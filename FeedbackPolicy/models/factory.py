import torch
import open_clip
from transformers import CLIPTextModel, CLIPTokenizer

from visual_planner.trainer import GoalGaussianDiffusion
from visual_planner.visual_planner import VisualPlanner

from ema_pytorch import EMA

from .policy import FeedbackDrivenPolicy
from .vit import VisionTransformer


def load_model(
    clip_vision_encoder_path: str,
    clip_vision_encoder_pretrained: str,
    target_size=(128, 128),
    use_vae=False,
    with_depth=True,
    flow_reg=True,
    sample_per_seq=8,
    diffusion_steps=100,
    sample_steps=20,
):
    # CLIP text tokenizer / encoder
    pretrained_model = "clip-vit-large-patch14"
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_name_or_path = pretrained_model)
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model_name_or_path = pretrained_model)
    text_encoder.requires_grad_(False)
    text_encoder.eval()


    # Visual planner
    unet = VisualPlanner(  
                image_size = target_size[0], 
                in_channels = 8,    # RGBD 
                out_channels = 4, 
                use_vae = use_vae,
                decoupled_output = False,
                temporal_length = sample_per_seq,
                dims = 3,
                flow_reg = flow_reg,
                with_state_estimate = False,
                )
    
    visual_planner = GoalGaussianDiffusion(
            channels=3,
            model=unet,
            image_size=target_size,
            timesteps=diffusion_steps,
            sampling_timesteps=sample_steps,
            loss_type='l2',
            objective='pred_v',
            beta_schedule = 'cosine',
            min_snr_loss_weight = True,
            auto_normalize = False,
            with_depth=with_depth,
            use_vae=use_vae,
        )
    visual_planner.eval()
    visual_planner = EMA(visual_planner, beta = 0.999, update_every = 10)


    # Policy with VC-1 as RGB encoder
    from vc_models.models.vit import model_utils 
    vision_encoder = model_utils.load_model(model_utils.VC1_BASE_NAME)
    embd_size = 768
    policy_model = FeedbackDrivenPolicy(
                        vision_encoder = vision_encoder, 
                        vis_dim = embd_size,  # 1024 for Large, 384 for Small
                        window_size = 5,
                        sampling_step = 1
    )
    policy_model.eval()


    return visual_planner, policy_model, tokenizer, text_encoder
