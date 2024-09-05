import math
import copy
from pathlib import Path
from random import random
from functools import partial
from collections import namedtuple
from multiprocessing import cpu_count

import torch
from torch import nn, einsum
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torch.optim import Adam, AdamW
from torchvision import transforms as T, utils

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange

from PIL import Image
from tqdm.auto import tqdm
from ema_pytorch import EMA
import imageio


from accelerate import Accelerator
from pytorch_fid.inception import InceptionV3
from pytorch_fid.fid_score import calculate_frechet_distance

import matplotlib.pyplot as plt
import numpy as np
import random

from diffusers.models import AutoencoderKL
from raft_utils.utils import flow_warp, robust_l1

__version__ = "0.0"

import os

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True


from pynvml import *

def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")

# constants
ModelPrediction =  namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])

# helpers functions
def tensors2vectors(tensors):
    def tensor2vector(tensor):
        flo = (tensor.permute(1, 2, 0).numpy()-0.5)*1000
        r = 8
        plt.quiver(flo[::-r, ::r, 0], -flo[::-r, ::r, 1], color='r', scale=r*20)
        plt.savefig('temp.jpg')
        plt.clf()
        return plt.imread('temp.jpg').transpose(2, 0, 1)
    return torch.from_numpy(np.array([tensor2vector(tensor) for tensor in tensors])) / 255

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def identity(t, *args, **kwargs):
    return t

def cycle(dl):
    while True:
        for data in dl:
            yield data

def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def convert_image_to_fn(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image

# normalization functions

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5


# gaussian diffusion trainer class

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def linear_beta_schedule(timesteps):
    """
    linear schedule, proposed in original ddpm paper
    """
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

def sigmoid_beta_schedule(timesteps, start = -3, end = 3, tau = 1, clamp_min = 1e-5):
    """
    sigmoid schedule
    proposed in https://arxiv.org/abs/2212.11972 - Figure 8
    better for images > 64x64, when used during training
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


   
class GoalGaussianDiffusion(nn.Module):
    def __init__(
        self,
        model,
        *,
        image_size,
        channels=3,
        timesteps = 1000,
        sampling_timesteps = 100,
        loss_type = 'l1',
        objective = 'pred_noise',
        beta_schedule = 'sigmoid',
        schedule_fn_kwargs = dict(),
        ddim_sampling_eta = 0.,
        auto_normalize = True,
        min_snr_loss_weight = False, # https://arxiv.org/abs/2303.09556
        min_snr_gamma = 5,
        with_depth = False,
        use_vae = False,
    ):
        super().__init__()
        # assert not (type(self) == GoalGaussianDiffusion and model.channels != model.out_dim)
        # assert not model.random_or_learned_sinusoidal_cond

        self.model = model
        self.channels = channels
        self.image_size = image_size
        self.objective = objective
        self.with_depth = with_depth
        self.use_vae = use_vae

        assert objective in {'pred_noise', 'pred_x0', 'pred_v'}, 'objective must be either pred_noise (predict noise) or pred_x0 (predict image start) or pred_v (predict v [v-parameterization as defined in appendix D of progressive distillation paper, used in imagen-video successfully])'

        if beta_schedule == 'linear':
            beta_schedule_fn = linear_beta_schedule
        elif beta_schedule == 'cosine':
            beta_schedule_fn = cosine_beta_schedule
        elif beta_schedule == 'sigmoid':
            beta_schedule_fn = sigmoid_beta_schedule
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        betas = beta_schedule_fn(timesteps, **schedule_fn_kwargs)

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        # sampling related parameters

        self.sampling_timesteps = default(sampling_timesteps, timesteps) # default num sampling timesteps to number of timesteps at training

        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        # helper function to register buffer from float64 to float32

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # derive loss weight
        # snr - signal noise ratio

        snr = alphas_cumprod / (1 - alphas_cumprod)

        # https://arxiv.org/abs/2303.09556

        maybe_clipped_snr = snr.clone()
        if min_snr_loss_weight:
            maybe_clipped_snr.clamp_(max = min_snr_gamma)

        if objective == 'pred_noise':
            register_buffer('loss_weight', maybe_clipped_snr / snr)
        elif objective == 'pred_x0':
            register_buffer('loss_weight', maybe_clipped_snr)
        elif objective == 'pred_v':
            register_buffer('loss_weight', maybe_clipped_snr / (snr + 1))

        # auto-normalization of data [0, 1] -> [-1, 1] - can turn off by setting it to be False

        self.normalize = normalize_to_neg_one_to_one if auto_normalize else identity
        self.unnormalize = unnormalize_to_zero_to_one if auto_normalize else identity

        self.negtive_prompt_cfg = None

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def predict_v(self, x_start, t, noise):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    def predict_start_from_v(self, x_t, t, v):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(self, x, t, x_cond, task_embed, state=None, clip_x_start=False, rederive_pred_noise=False, guidance_weight=0, negtive_prompt_cfg=None):

        model_output = self.model(x, t, text_embedding = task_embed, x_cond = x_cond, state = state, forward=False)
        if guidance_weight > 0.0:
            uncond_model_output = self.model(x, t, text_embedding = task_embed * 0.0, x_cond = x_cond, state = state, forward=False)

        maybe_clip = partial(torch.clamp, min = -1., max = 1.) if clip_x_start else identity

        if self.objective == 'pred_noise':
            if guidance_weight == 0:
                pred_noise = model_output
            else:
                pred_noise = (1 + guidance_weight)*model_output - guidance_weight*uncond_model_output

            x_start = self.predict_start_from_noise(x, t, pred_noise)
            x_start = maybe_clip(x_start)

            if clip_x_start and rederive_pred_noise:
                pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_x0':
            x_start = model_output
            x_start = maybe_clip(x_start)

            if guidance_weight == 0:
                pred_noise = self.predict_noise_from_start(x, t, x_start)
            else:
                uncond_x_start = uncond_model_output
                uncond_x_start = maybe_clip(uncond_x_start)
                cond_noise = self.predict_noise_from_start(x, t, x_start)
                uncond_noise = self.predict_noise_from_start(x, t, uncond_x_start)
                pred_noise = (1 + guidance_weight)*cond_noise - guidance_weight*uncond_noise
                x_start = self.predict_start_from_noise(x, t, pred_noise)
            
        elif self.objective == 'pred_v':
            v = model_output
            x_start = self.predict_start_from_v(x, t, v)
            x_start = maybe_clip(x_start)
            
            if guidance_weight == 0:
                pred_noise = self.predict_noise_from_start(x, t, x_start)
            else:
                uncond_v = uncond_model_output
                uncond_x_start = self.predict_start_from_v(x, t, uncond_v)
                uncond_noise = self.predict_noise_from_start(x, t, uncond_x_start)
                cond_noise = self.predict_noise_from_start(x, t, x_start)
                pred_noise = (1 + guidance_weight)*cond_noise - guidance_weight*uncond_noise
                x_start = self.predict_start_from_noise(x, t, pred_noise)

        return ModelPrediction(pred_noise, x_start)

    def p_mean_variance(self, x, t, x_cond, task_embed,  clip_denoised=False, guidance_weight=0, negtive_prompt_cfg=None):
        preds = self.model_predictions(x, t, x_cond, task_embed, guidance_weight=guidance_weight, negtive_prompt_cfg=negtive_prompt_cfg)
        x_start = preds.pred_x_start

        if clip_denoised:
            x_start.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start = x_start, x_t = x, t = t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.no_grad()
    def p_sample(self, x, t: int, x_cond, task_embed, guidance_weight=0, negtive_prompt_cfg=None):
        b, *_, device = *x.shape, x.device
        batched_times = torch.full((b,), t, device = x.device, dtype = torch.long)

        model_mean, _, model_log_variance, x_start = self.p_mean_variance(x, batched_times, x_cond, task_embed\
        , clip_denoised = True, guidance_weight=guidance_weight, negtive_prompt_cfg=negtive_prompt_cfg)

        noise = torch.randn_like(x) if t > 0 else 0. # no noise if t == 0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start

    @torch.no_grad()
    def p_sample_loop(self, shape, x_cond, task_embed, return_all_timesteps=False, guidance_weight=0, negtive_prompt_cfg=None):
        batch, device = shape[0], self.betas.device

        img = torch.randn(shape, device=device)
        if self.with_depth and not self.use_vae:
            b, f, c, w, h = shape
            img = torch.cat([torch.randn((b,f,3,h,w), device=device), torch.randn((b,f,1,h,w), device=device)], dim=2)
        imgs = [img]

        x_start = None

        for t in tqdm(reversed(range(0, self.num_timesteps)), desc = 'sampling loop time step', total = self.num_timesteps):
            # self_cond = x_start if self.self_condition else None

            img, x_start = self.p_sample(img, t, x_cond, task_embed, guidance_weight=guidance_weight, negtive_prompt_cfg=negtive_prompt_cfg)
            imgs.append(img)

        ret = img if not return_all_timesteps else torch.stack(imgs, dim = 1)

        ret = self.unnormalize(ret)
        return ret

    @torch.no_grad()
    def ddim_sample(self, shape, x_cond, task_embed, state=None, return_all_timesteps=False, guidance_weight=0, negtive_prompt_cfg=None):
        batch, device, total_timesteps, sampling_timesteps, eta, objective = shape[0], self.betas.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

        times = torch.linspace(-1, total_timesteps - 1, steps = sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        img = torch.randn(shape, device=device)
        if self.with_depth and not self.use_vae:
            b, f, c, w, h = shape
            img = torch.cat([torch.randn((b,f,3,h,w), device=device), torch.randn((b,f,1,h,w), device=device)], dim=2)
        imgs = [img]

        x_start = None

        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
            time_cond = torch.full((batch,), time, device = device, dtype = torch.long)
            # self_cond = x_start if self.self_condition else None
            pred_noise, x_start, *_ = self.model_predictions(img, time_cond, x_cond, task_embed, state=state, clip_x_start = False, \
                                      rederive_pred_noise = True, guidance_weight=guidance_weight, negtive_prompt_cfg=negtive_prompt_cfg)

            if time_next < 0:
                img = x_start
                imgs.append(img)
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(img)

            img = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise

            imgs.append(img)

        ret = img if not return_all_timesteps else torch.stack(imgs, dim = 1)

        ret = self.unnormalize(ret)
        return ret

    @torch.no_grad()
    def sample(self, x_cond, task_embed, batch_size = 16, return_all_timesteps = False, guidance_weight=0, frames=16, state=None, negtive_prompt_cfg=None):
        image_size, channels = self.image_size, self.channels
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        return sample_fn((batch_size, frames, channels, image_size[0], image_size[1]), x_cond, task_embed,  return_all_timesteps = return_all_timesteps, guidance_weight=guidance_weight, state=state, negtive_prompt_cfg=negtive_prompt_cfg)

    @torch.no_grad()
    def interpolate(self, x1, x2, t = None, lam = 0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.full((b,), t, device = device)
        xt1, xt2 = map(lambda x: self.q_sample(x, t = t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2

        x_start = None

        for i in tqdm(reversed(range(0, t)), desc = 'interpolation sample time step', total = t):
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(img, i, self_cond)

        return img

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    @property
    def loss_fn(self):
        if self.loss_type == 'l1':
            return F.l1_loss
        elif self.loss_type == 'l2':
            return F.mse_loss
        else:
            raise ValueError(f'invalid loss type {self.loss_type}')

    def compute_flow_reg(self, x_start, flow_out):
        flow_loss = 0
        for i in range(len(flow_out)):
            wraped_imgs = flow_warp(x_start[:, i, :3], flow_out[i])

            # visibility mask
            valid_pixels = 1 - (wraped_imgs == 0).prod(1, keepdim=True).type_as(wraped_imgs)
            diff = (x_start[:, i+1, :3] - wraped_imgs) * valid_pixels
            identity_diff = (x_start[:, i+1, :3] - x_start[:, i, :3]) * valid_pixels

            # movement mask
            motion_mask = torch.where(diff < identity_diff, 1, 0)

            # compute opentical flow regularization
            flow_loss += robust_l1(diff * motion_mask) / len(flow_out)

        return flow_loss
    

    def p_losses(self, x_start, t, x_cond, task_embed, state, noise=None):
        b, f, c, h, w = x_start.shape
        noise = default(noise, lambda: torch.randn_like(x_start))
        if self.with_depth and not self.use_vae:
            # decoupled noise for RGB and depth
            noise = torch.cat([torch.randn_like(x_start[:,:,:3]), torch.randn_like(x_start[:,:,3:])], dim=2)

        # get noised sample
        x = self.q_sample(x_start=x_start, t=t, noise=noise)

        # predict and take gradient step
        if self.model.flow_reg:
            model_out, flow_out = self.model(x, t, text_embedding = task_embed, x_cond = x_cond, x_start=x_start, forward=True)
        else:
            model_out = self.model(x, t, text_embedding = task_embed, x_cond = x_cond, x_start=x_start, forward=True)
        
        flow_loss = self.compute_flow_reg(x_start, flow_out) if self.model.flow_reg else torch.tensor(0)


        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start
        elif self.objective == 'pred_v':
            v = self.predict_v(x_start, t, noise)
            target = v
        else:
            raise ValueError(f'unknown objective {self.objective}')

        # compute diffsuion loss
        loss = self.loss_fn(model_out, target, reduction = 'none')
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss * extract(self.loss_weight, t, loss.shape)

        return loss.mean() + flow_loss, {'Flow': flow_loss}

    def forward(self, img, img_cond, task_embed, state):
        b, f, c, h, w, device, img_size, = *img.shape, img.device, self.image_size
        # assert h == img_size[0] and w == img_size[1], f'height and width of image must be {img_size}, got({h}, {w})'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        img = self.normalize(img)
        return self.p_losses(img, t, img_cond, task_embed, state)


# trainer class
class Trainer(object):
    def __init__(
        self,
        args,
        model,
        diffusion,
        latent_size,
        image_size,
        tokenizer, 
        text_encoder, 
        train_set,
        valid_set,
        channels = 3,
        *,
        train_batch_size = 1,
        valid_batch_size = 1,
        gradient_accumulate_every = 1,
        augment_horizontal_flip = True,
        train_lr = 1e-4,
        train_num_steps = 100000,
        ema_update_every = 10,
        ema_decay = 0.995,
        adam_betas = (0.9, 0.99),
        save_and_sample_every = 1000,
        num_samples = 3,
        results_folder = './results',
        amp = True,
        fp16 = True,
        split_batches = True,
        convert_image_to = None,
        cond_drop_chance=0.1,
        wandb = None,
        use_vae = True,
        with_depth = False,
    ):
        super().__init__()
        self.args = args
        self.cond_drop_chance = cond_drop_chance

        self.tokenizer = tokenizer
        self.text_encoder = text_encoder

        # accelerator
        self.accelerator = Accelerator(
            split_batches = split_batches,
            mixed_precision = 'fp16' if fp16 else 'no'
        )
        self.accelerator.native_amp = amp


        # model
        self.model = model
        self.diffusion = diffusion
        self.latent_size = latent_size
        self.channels = channels

        self.use_vae = use_vae
        self.with_depth = with_depth
        if self.use_vae:
            if self.with_depth:
                self.vae = AutoencoderKL.from_pretrained("Intel/ldm3d-4c", subfolder="vae").to('cuda')
                self.vae.requires_grad_(False)
            else:
                self.vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", subfolder="vae").to('cuda')
                self.vae.requires_grad_(False)
        self.with_text_conditioning = args.with_text_conditioning
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
        self.inception_v3 = InceptionV3([block_idx])
        self.inception_v3.to(self.device)


        # sampling and training hyperparameters
        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every
        self.batch_size = train_batch_size
        self.valid_batch_size = valid_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.train_num_steps = train_num_steps
        self.image_size = image_size
        self.fp16 = fp16


        # dataset and dataloader
        self.ds = train_set
        self.valid_ds = valid_set
        dl = DataLoader(self.ds, batch_size = train_batch_size, shuffle = True, pin_memory = True, num_workers = 2)


        dl = self.accelerator.prepare(dl)
        self.dl = cycle(dl)
        self.valid_dl = DataLoader(self.valid_ds, batch_size = valid_batch_size, shuffle = True, pin_memory = True, num_workers = 1)

        # optimizer
        self.opt = AdamW(self.model.parameters(), lr = train_lr, betas = adam_betas, weight_decay = 0.)


        if self.accelerator.is_main_process:
            self.ema = EMA(self.model, beta = ema_decay, update_every = ema_update_every)
            self.ema.to(self.device)

        # for logging results in a folder periodically
        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok = True, parents = True)

        # step counter state
        self.step = 0

        # prepare model, dataloader, optimizer with accelerator
        self.model, self.opt, self.text_encoder = \
            self.accelerator.prepare(self.model, self.opt, self.text_encoder)


        self.wandb = wandb


    @property
    def device(self):
        return self.accelerator.device

    def save(self, milestone):
        if not self.accelerator.is_local_main_process:
            return

        data = {
            'step': self.step,
            'model': self.accelerator.get_state_dict(self.model),
            'opt': self.opt.state_dict(),
            'ema': self.ema.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None,
            'version': __version__
        }

        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))

    def load(self, milestone):
        accelerator = self.accelerator
        device = accelerator.device

        data = torch.load(str(self.results_folder / f'model-{milestone}.pt'), map_location=device)

        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data['model'])

        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        if self.accelerator.is_main_process:
            self.ema.load_state_dict(data["ema"])

        if 'version' in data:
            print(f"loading from version {data['version']}")

        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])

    # return fid_value
    def encode_batch_text(self, batch_text):
        batch_text_ids = self.tokenizer(batch_text, return_tensors = 'pt', padding = True, truncation = True, max_length = 32).to(self.device)
        batch_text_embed = self.text_encoder(**batch_text_ids).last_hidden_state
        return batch_text_embed

    def sample(self, x_conds, batch_text, batch_size=1, guidance_weight=0):
        device = self.device
        task_embeds = self.encode_batch_text(batch_text)
        return self.ema.ema_model.sample(x_conds.to(device), task_embeds.to(device), batch_size=batch_size, guidance_weight=guidance_weight)

    def train(self):
        accelerator = self.accelerator
        device = accelerator.device

        with tqdm(initial = self.step, total = self.train_num_steps, disable = not accelerator.is_main_process) as pbar:

            while self.step < self.train_num_steps:

                total_loss = 0.

                for _ in range(self.gradient_accumulate_every):
                    x, x_cond, goal, state = next(self.dl)
                    x, x_cond = x.to(device), x_cond.to(device)
                    state = state.to(device)

                    if self.use_vae:
                        with torch.no_grad():
                            b = x.shape[0]
                            x = torch.cat([x_cond, x], dim = 1)
                            x = rearrange(x, 'b f c h w -> (b f) c h w').contiguous()
                            x = self.vae.encode(x).latent_dist.sample().mul_(0.18215)
                            x = rearrange(x, '(b f) c h w -> b f c h w', b=b).contiguous()

                            x_cond = x[:, :1]
                            x = x[:, 1:]


                    if self.with_text_conditioning:
                        goal_embed = self.encode_batch_text(goal)
                        goal_embed = goal_embed * (torch.rand(goal_embed.shape[0], 1, 1, device = goal_embed.device) > self.cond_drop_chance).float()
                    else:
                        goal_embed = None


                    with self.accelerator.autocast():

                        if self.use_vae:
                            model_kwargs = {'text_embedding':goal_embed, 'x_cond':x_cond, 'state':state}
                            t = torch.randint(0, self.diffusion.num_timesteps, (x.shape[0],), device=device)
                            loss_dict = self.diffusion.training_losses(self.model, x, t, model_kwargs)
                            loss = loss_dict["loss"].mean() / self.gradient_accumulate_every

                        else:
                            loss, loss_dict = self.model(x, x_cond, goal_embed, state)
                            loss = loss / self.gradient_accumulate_every
                        
                        total_loss += loss.item()
                        self.accelerator.backward(loss)
                
                    if self.args.rank == 0 and self.args.report_to_wandb:
                        self.wandb.log(
                            {
                                "loss": loss.item(),
                                "Flow": loss_dict['Flow'].item(),
                            },
                            commit=True,
                        )

                accelerator.clip_grad_norm_(self.model.parameters(), 0.1)

                try:
                    scale = self.accelerator.scaler.get_scale()
                except:
                    scale = .0
                flow_loss = loss_dict['Flow'].item()
                pbar.set_description(f'loss: {total_loss:.4E} Flow: {flow_loss:.4E} loss scale: {scale:.1E}')

                accelerator.wait_for_everyone()

                self.opt.step()
                self.opt.zero_grad()

                accelerator.wait_for_everyone()

                self.step += 1
                if accelerator.is_main_process:
                    self.ema.update()

                    if self.step != 0 and self.step % self.save_and_sample_every == 0:
                        self.ema.ema_model.eval()

                        with torch.no_grad():
                            milestone = self.step // self.save_and_sample_every
                            batches = num_to_groups(self.num_samples, self.valid_batch_size)
                            ### get val_imgs from self.valid_dl
                            x_conds = []
                            xs = []
                            labels = []
                            task_embeds = []
                            for i, (x, x_cond, label, state) in enumerate(self.valid_dl):

                                if i >= self.num_samples:
                                    break
                                
                                x_cond = x_cond.to(device)
                                state = state.to(device)
                                if self.use_vae:
                                    b = x_cond.shape[0]
                                    x_cond = rearrange(x_cond, 'b f c h w -> (b f) c h w').contiguous()
                                    x_cond = self.vae.encode(x_cond).latent_dist.sample().mul_(0.18215)
                                    x_cond = rearrange(x_cond, '(b f) c h w -> b f c h w', b=b).contiguous()

                                x_conds.append(x_cond)
                                xs.append(x)
                                labels.append(label)
                                task_embed = self.encode_batch_text(label)
                                task_embeds.append(task_embed)
                            
                            with self.accelerator.autocast():
                                if self.use_vae:
                                    if self.fp16:
                                        z = torch.randn(len(x_conds), self.args.sample_per_seq, 4, self.latent_size, self.latent_size, dtype=torch.float16, device=device)
                                    else:
                                        z = torch.randn(len(x_conds), self.args.sample_per_seq, 4, self.latent_size, self.latent_size, device=device)

                                    all_xs_list = list(map(lambda n, c, e: self.diffusion.p_sample_loop(
                                        self.ema.ema_model.forward, z.shape, z, clip_denoised=True, model_kwargs={'text_embedding':e, 'x_cond':c, 'state':state}, progress=True, device=device
                                    ), batches, x_conds, task_embeds))

                                else:
                                    all_xs_list = list(map(lambda n, c, e: self.ema.ema_model.sample(batch_size=n, x_cond=c, task_embed=e, state=state, frames=self.args.sample_per_seq, guidance_weight=0), batches, x_conds, task_embeds))
                                

                        
                        print_gpu_utilization()
                        
                        gt_xs = torch.cat(xs, dim = 0).to(device)
                        all_xs = torch.cat(all_xs_list, dim = 0).detach()#.cpu()

                        if self.use_vae:
                            b = all_xs.shape[0]
                            all_xs = rearrange(all_xs, 'b f c h w -> (b f) c h w')
                            all_xs = self.vae.decode(all_xs / 0.18215).sample 
                            all_xs = rearrange(all_xs, '(b f) c h w -> b f c h w', b=b)

                        gt_img = (gt_xs * 0.5 + 0.5).clamp(0, 1)
                        pred_img = (all_xs * 0.5 + 0.5).clamp(0, 1)

    
                        # Results Visualization
                        n_rows = 4          # Number of frames per row
                        fps = 4             # Frames per second
                        print('Evaluated Task: ', labels[0])
                        os.makedirs(str(self.results_folder / f'imgs'), exist_ok = True)
                        for i in range(len(gt_xs)):
                            if self.with_depth:
                                # Visualize RGB
                                gt_img_ = gt_img[i, :, :3].cpu()     # rearrange(gt_img, 'b n c h w -> (b n) c h w')
                                utils.save_image(gt_img_, str(self.results_folder / f'imgs/gt_img_{milestone}_{i}.png'), nrow=n_rows)
                                # Visualize Depth
                                gt_img_ = gt_img[i, :, 3:].cpu()     # rearrange(gt_img, 'b n c h w -> (b n) c h w')
                                utils.save_image(gt_img_, str(self.results_folder / f'imgs/gt_img_depth_{milestone}_{i}.png'), nrow=n_rows)
                            else:
                                gt_img_ = gt_img[i].cpu()     # rearrange(gt_img, 'b n c h w -> (b n) c h w')
                                utils.save_image(gt_img_, str(self.results_folder / f'imgs/gt_img_{milestone}_{i}.png'), nrow=n_rows)

                        os.makedirs(str(self.results_folder / f'imgs/outputs'), exist_ok = True)

                        
                        for i in range(len(gt_xs)):
                            if self.with_depth:
                                    # Visualize RGB
                                    pred_img_ = pred_img[i, :, :3].cpu()     # rearrange(gt_img, 'b n c h w -> (b n) c h w')
                                    utils.save_image(pred_img_, str(self.results_folder / f'imgs/sample_{milestone}_{i}.png'), nrow=n_rows)

                                    video_ = (pred_img_ * 255).add_(0.5).clamp_(0, 255).to(dtype=torch.uint8).permute(0, 2, 3, 1).contiguous()
                                    video_save_path = os.path.join(self.results_folder, f'imgs/sample_{milestone}_{i}' + '.mp4')
                                    imageio.mimwrite(video_save_path, video_, fps=fps, quality=9)

                                    # Visualize Depth
                                    pred_img_ = pred_img[i, :, 3:].cpu()      # rearrange(gt_img, 'b n c h w -> (b n) c h w')
                                    utils.save_image(pred_img_, str(self.results_folder / f'imgs/sample_depth_{milestone}_{i}.png'), nrow=n_rows)

                                    video_ = (pred_img_ * 255).add_(0.5).clamp_(0, 255).to(dtype=torch.uint8).permute(0, 2, 3, 1).contiguous()
                                    video_save_path = os.path.join(self.results_folder, f'imgs/sample_depth_{milestone}_{i}' + '.mp4')
                                    imageio.mimwrite(video_save_path, video_, fps=fps, quality=9)
                            else:
                                pred_img_ = pred_img[i].cpu()     # rearrange(pred_img, 'b n c h w -> (b n) c h w')
                                utils.save_image(pred_img_, str(self.results_folder / f'imgs/outputs/sample-{milestone}-{i}.png'), nrow=n_rows)

                                video_ = (pred_img_ * 255).add_(0.5).clamp_(0, 255).to(dtype=torch.uint8).permute(0, 2, 3, 1).contiguous()
                                video_save_path = os.path.join(self.results_folder, f'imgs/sample_{milestone}_{i}' + '.mp4')
                                imageio.mimwrite(video_save_path, video_, fps=fps, quality=9)


                        self.save(milestone)

                pbar.update(1)

        accelerator.print('training complete')


    def eval(self,):
        from metric_utils import calculate_ssim, calculate_psnr, calculate_lpips

        # Setup seed for reproducing
        setup_seed(123)

        # Track Activation Statics from Inception
        act_pred_list = []
        act_gt_list = []
        metrics = {'SSIM':0, 'PSNR':0, 'LPIPS':0, 'RMSE':0 }
        tasks = []
        device = self.device
        
        with torch.no_grad():
            milestone = self.step // self.save_and_sample_every
            batches = 2048 // (self.args.val_batch_size * self.args.sample_per_seq)    

            for i, (x, x_cond, label, state) in enumerate(self.valid_dl):
                if i >= batches:
                    break
                

                x = x.to(device)
                x_cond = x_cond.to(device)
                if self.use_vae:
                    b = x_cond.shape[0]
                    x_cond = rearrange(x_cond, 'b f c h w -> (b f) c h w').contiguous()
                    x_cond = self.vae.encode(x_cond).latent_dist.sample().mul_(0.18215)
                    x_cond = rearrange(x_cond, '(b f) c h w -> b f c h w', b=b).contiguous()

                task = label
                tasks += task
                task_embed = self.encode_batch_text(task)
                with self.accelerator.autocast():
                    pred_img = self.ema.ema_model.sample(batch_size=x.shape[0], x_cond=x_cond, task_embed=task_embed, state=state, \
                               frames=self.args.sample_per_seq, guidance_weight=0, negtive_prompt_cfg=self.negtive_prompt_embed)

                
                print_gpu_utilization()

                pred_img = (pred_img * 0.5 + 0.5).clamp(0, 1)
                gt_img = (x * 0.5 + 0.5).clamp(0, 1)

                
                if self.use_vae:
                    b = pred_img.shape[0]
                    pred_img = rearrange(pred_img, 'b f c h w -> (b f) c h w')
                    pred_img = self.vae.decode(pred_img / 0.18215).sample 
                    pred_img = rearrange(pred_img, '(b f) c h w -> b f c h w', b=b)
                


                lpips_results = calculate_lpips(pred_img[:, :, :3], gt_img[:, :, :3], device=device)
                ssim_results  = calculate_ssim(pred_img[:, :, :3].cpu(), gt_img[:, :, :3].cpu())
                psnr_results  = calculate_psnr(pred_img[:, :, :3].cpu(), gt_img[:, :, :3].cpu())
                
                act_pred = self.inception_v3(pred_img[:, :, :3].flatten(0, 1))[0].squeeze().cpu()#.numpy()
                act_gt = self.inception_v3(gt_img[:, :, :3].flatten(0, 1))[0].squeeze().cpu()#.numpy()

                act_pred_list.append(act_pred)
                act_gt_list.append(act_gt)

                if self.with_depth:
                    metrics['RMSE'] += F.mse_loss(pred_img[:, :, 3:], gt_img[:, :, 3:]).sqrt().item() / batches
                metrics['PSNR'] += np.mean([value for value in psnr_results['value'].values()]) / batches
                metrics['SSIM'] += np.mean([value for value in ssim_results['value'].values()]) / batches
                metrics['LPIPS'] += np.mean([value for value in lpips_results['value'].values()]) / batches
                print(f'Batches {i+1} / {batches} compelted')

            
            act_pred = torch.cat(act_pred_list, dim=0).numpy()
            gt_pred = torch.cat(act_gt_list, dim=0).numpy()
            mu1 = np.mean(act_pred, axis=0)
            sigma1 = np.cov(act_pred, rowvar=False)
            mu2 = np.mean(gt_pred, axis=0)
            sigma2 = np.cov(gt_pred, rowvar=False)
            fid = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
            metrics['FID'] = fid
            print('Evaluation Results: ', metrics)


            import json 
            # Convert and write JSON object to file
            checkpoint_num = int(self.args.resume_path.split('/')[-1].split('-')[-1].split('.')[0])
            with open(str(self.results_folder / f'evaluation_metrics_{checkpoint_num}.json'), "w") as outfile: 
                metrics['Evaluated Tasks'] = tasks
                json.dump(metrics, outfile, indent=4)


            # Results Visualization
            os.makedirs(str(self.results_folder / f'imgs'), exist_ok = True)
            fps = 4
            n_rows = 8

            for i in range(len(gt_img[:16])):
                if self.with_depth:
                    # Visualize RGB
                    gt_img_ = gt_img[i, :, :3].cpu()     # rearrange(gt_img, 'b n c h w -> (b n) c h w')
                    utils.save_image(gt_img_, str(self.results_folder / f'imgs/gt_img_{i}.png'), nrow=n_rows)
                    # Visualize Depth
                    gt_img_ = gt_img[i, :, 3:].cpu()     # rearrange(gt_img, 'b n c h w -> (b n) c h w')
                    utils.save_image(gt_img_, str(self.results_folder / f'imgs/gt_img_depth_{i}.png'), nrow=n_rows)
                else:
                    gt_img_ = gt_img[i].cpu()     # rearrange(gt_img, 'b n c h w -> (b n) c h w')
                    utils.save_image(gt_img_, str(self.results_folder / f'imgs/gt_img_{i}.png'), nrow=n_rows)

            for i in range(len(pred_img[:16])):
                if self.with_depth:
                        # Visualize RGB
                        pred_img_ = pred_img[i, :, :3].cpu()     # rearrange(gt_img, 'b n c h w -> (b n) c h w')
                        utils.save_image(pred_img_, str(self.results_folder / f'imgs/sample_{i}.png'), nrow=n_rows)

                        video_ = (pred_img_ * 255).add_(0.5).clamp_(0, 255).to(dtype=torch.uint8).permute(0, 2, 3, 1).contiguous()
                        video_save_path = os.path.join(self.results_folder, f'imgs/sample_{i}' + '.mp4')
                        imageio.mimwrite(video_save_path, video_, fps=fps, quality=9)

                        # Visualize Depth
                        pred_img_ = pred_img[i, :, 3:].cpu()      # rearrange(gt_img, 'b n c h w -> (b n) c h w')
                        utils.save_image(pred_img_, str(self.results_folder / f'imgs/sample_depth_{i}.png'), nrow=n_rows)

                        video_ = (pred_img_ * 255).add_(0.5).clamp_(0, 255).to(dtype=torch.uint8).permute(0, 2, 3, 1).contiguous()
                        video_save_path = os.path.join(self.results_folder, f'imgs/sample_depth_{i}' + '.mp4')
                        imageio.mimwrite(video_save_path, video_, fps=fps, quality=9)
                else:
                    pred_img_ = pred_img[i].cpu()     # rearrange(pred_img, 'b n c h w -> (b n) c h w')
                    utils.save_image(pred_img_, str(self.results_folder / f'imgs/sample-{i}.png'), nrow=n_rows)

                    video_ = (pred_img_ * 255).add_(0.5).clamp_(0, 255).to(dtype=torch.uint8).permute(0, 2, 3, 1).contiguous()
                    video_save_path = os.path.join(self.results_folder, f'imgs/sample_{i}' + '.mp4')
                    imageio.mimwrite(video_save_path, video_, fps=fps, quality=9)


