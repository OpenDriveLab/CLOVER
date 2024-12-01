import os
import torch
from transformers import CLIPTextModel, CLIPTokenizer

from visual_planner.trainer import GoalGaussianDiffusion
from visual_planner.visual_planner import VisualPlanner

from ema_pytorch import EMA

from .policy import FeedbackDrivenPolicy
from .vit import VisionTransformer


IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

IMAGENET_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
IMAGENET_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


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



def create_feedback_policy(
    vision_encoder: str = 'vc1-base',   #TODO: Support additional visual encoders
    resume_from_checkpoint: str = None,
):

    import torchvision.transforms as transforms
    image_processor = transforms.Compose([
                            transforms.Resize((192, 192), interpolation = transforms.InterpolationMode.BICUBIC),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
                    ])
    pretrained_model = "clip-vit-large-patch14"
    text_tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_name_or_path = pretrained_model)

    from vc_models.models.vit import model_utils
    vision_encoder = model_utils.load_model(model_utils.VC1_BASE_NAME)
    embd_size = 768


    model = FeedbackDrivenPolicy(vision_encoder = vision_encoder, \
            vis_dim = embd_size,
            window_size = 5,
            sampling_step = 1)
    
    model.vision_encoder.requires_grad_(False)

    def check_file_exists(file_path):
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"The file '{file_path}' does not exist.")


    print('Try loading from ckpt')
    try:
        check_file_exists(resume_from_checkpoint)
        old_ckpt = torch.load(resume_from_checkpoint)['model_state_dict']

        # remove 'module.' in original keys
        new_ckpt = {}
        for k, v in old_ckpt.items():
            new_ckpt[k[7:]] = v
        model.load_state_dict(new_ckpt, strict=False)

    except FileNotFoundError as e:
        print(e)
 
    return model, image_processor, text_tokenizer