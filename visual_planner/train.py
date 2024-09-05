import os
import torch
import wandb
import argparse

# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from trainer import GoalGaussianDiffusion, Trainer
from visual_planner import VisualPlanner
from transformers import CLIPTextModel, CLIPTokenizer

from calvin_data import DiskCalvinDataset
from pathlib import Path


def world_info_from_env():
    local_rank = 0
    for v in (
        "LOCAL_RANK",
        "MPI_LOCALRANKID",
        "SLURM_LOCALID",
        "OMPI_COMM_WORLD_LOCAL_RANK",
    ):
        if v in os.environ:
            local_rank = int(os.environ[v])
            break
    global_rank = 0
    for v in ("RANK", "PMI_RANK", "SLURM_PROCID", "OMPI_COMM_WORLD_RANK"):
        if v in os.environ:
            global_rank = int(os.environ[v])
            break
    world_size = 1
    for v in ("WORLD_SIZE", "PMI_SIZE", "SLURM_NTASKS", "OMPI_COMM_WORLD_SIZE"):
        if v in os.environ:
            world_size = int(os.environ[v])
            break

    return local_rank, global_rank, world_size


def main(args, wandb = None):

    target_size = (128, 128)
    sampling_step = args.sampling_step
    window_size = args.sample_per_seq * sampling_step + sampling_step
    train_set = DiskCalvinDataset(
        datasets_dir=Path('calvin/dataset/task_ABC_D') / "training",
        window_size=window_size,
        sampling_step=sampling_step,
        image_size=target_size[0],
        with_depth=args.with_depth,
    )

    valid_set = DiskCalvinDataset(
        datasets_dir=Path('calvin/dataset/task_ABC_D') / "validation",
        window_size=window_size,
        sampling_step=sampling_step,
        image_size=target_size[0],
        with_depth=args.with_depth,
    )

    
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_name_or_path = "openai/clip-vit-large-patch14")
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model_name_or_path = "openai/clip-vit-large-patch14")
    text_encoder.requires_grad_(False)
    text_encoder.eval()


    # We use UNet-based Diffusion model as the visual planner
    raw_input_channels = 8 if args.with_depth else 6
    model = VisualPlanner(  
                    image_size = target_size[0] // 8 if args.use_vae else target_size[0], 
                    in_channels = 8 if args.use_vae else raw_input_channels, 
                    out_channels = 8 if args.use_vae else raw_input_channels // 2, 
                    use_vae = args.use_vae,
                    decoupled_output = False,
                    temporal_length = args.sample_per_seq,          # Number of frames to predict
                    dims = 3,
                    flow_reg = args.flow_reg,
                    with_state_estimate = False,
                    )

    print(
        f"Model initialized with {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters"
    )

    if args.use_vae:
        from guided_diffusion.guided_diffusion import create_diffusion
        gaussian_diffusion = create_diffusion(timestep_respacing="", diffusion_steps=args.diffusion_steps)
    else:
        diffusion = GoalGaussianDiffusion(
            channels=3,
            model=model,
            image_size=target_size,
            timesteps=args.diffusion_steps,
            sampling_timesteps=args.sample_steps,
            loss_type='l2',
            objective='pred_v',
            beta_schedule = 'cosine',
            min_snr_loss_weight = True,
            auto_normalize = False,
            with_depth=args.with_depth,
            use_vae=args.use_vae,
        )

    if os.path.exists(args.resume_path):
        model_ckpt = torch.load(args.resume_path)['model']
        if args.use_vae:
            model.load_state_dict(model_ckpt)
        else:
            diffusion.load_state_dict(model_ckpt)   # Model warped with diffusion
        print('resume ckpt successfully loaded form: ', args.resume_path)


    trainer = Trainer(
        args = args,
        model=model if args.use_vae else diffusion,
        diffusion = gaussian_diffusion if args.use_vae else None,
        latent_size = target_size[0] // 8 if args.use_vae else None,
        image_size = target_size,
        tokenizer=tokenizer, 
        text_encoder=text_encoder,
        train_set=train_set,
        valid_set=valid_set,
        train_lr=args.learning_rate,
        train_num_steps = args.train_num_steps,
        save_and_sample_every = args.save_and_sample_every,
        ema_update_every = args.ema_update_every,
        ema_decay = args.ema_decay,
        train_batch_size = args.train_batch_size,
        valid_batch_size = args.val_batch_size,
        gradient_accumulate_every = args.gradient_accumulate_every,
        num_samples=1, 
        results_folder = args.results_folder,
        fp16 = False,
        amp = False,
        wandb = wandb,
        use_vae=args.use_vae,
        with_depth=args.with_depth,
        cond_drop_chance = 0.,     # Classifier free guidance
    )

    if os.path.exists(args.resume_path) and args.load_trainer:
        checkpoint_num = int(args.resume_path.split('/')[-1].split('-')[-1].split('.')[0])
        trainer.load(checkpoint_num)
    
    if args.mode == 'train':
        trainer.train()
    else:
        model.eval()
        diffusion.eval()
        trainer.eval()

        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', type=str, default='train', choices=['train', 'val']) # set to 'inference' to generate samples
    parser.add_argument('-c', '--checkpoint_num', type=int, default=None) # set to checkpoint number to resume training or generate samples
    parser.add_argument('-p', '--inference_path', type=str, default=None) # set to path to generate samples
    parser.add_argument('-t', '--text', type=str, default=None) # set to text to generate samples
    parser.add_argument('-n', '--sample_steps', type=int, default=100) # set to number of steps to sample
    parser.add_argument('-g', '--guidance_weight', type=int, default=0) # set to positive to use guidance

    # Training Config
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--train_num_steps', type=int, default=60000)
    parser.add_argument('--save_and_sample_every', type=int, default=2500)
    parser.add_argument('--train_batch_size', type=int, default=16)
    parser.add_argument('--val_batch_size', type=int, default=1)
    parser.add_argument('--gradient_accumulate_every', type=int, default=1)
    parser.add_argument('--sample_per_seq', type=int, default=8)
    parser.add_argument('--resume_path', type=str, default='')
    parser.add_argument('--load_trainer', default=False, action="store_true")
    parser.add_argument('--sampling_step', type=int, default=5)
    
    # EMA config
    parser.add_argument('--ema_update_every', type=int, default=10)
    parser.add_argument('--ema_decay', type=float, default=0.999)

    # Model Config
    parser.add_argument('--use_vae', default=False, action="store_true")
    parser.add_argument('--flow_reg', default=False, action="store_true")
    parser.add_argument('--with_depth', default=False, action="store_true")
    parser.add_argument('--with_text_conditioning', default=False, action="store_true")
    parser.add_argument('--diffusion_steps', type=int, default=100)

    # Log Config
    parser.add_argument('--report_to_wandb', default=False, action="store_true")
    parser.add_argument('--run_name', type=str, default='train_visual_planner')
    parser.add_argument('--results_folder', type=str, default='../results/visual_planner')
    
    args = parser.parse_args()
    args.local_rank, args.rank, args.world_size = world_info_from_env()

    if args.rank == 0 and args.report_to_wandb:
        wandb.init(
            name=args.run_name,
            config=vars(args),
        )

    if args.mode == 'inference':
        assert args.checkpoint_num is not None
        assert args.inference_path is not None
        assert args.text is not None
        assert args.sample_steps <= 100
    main(args, wandb=wandb)