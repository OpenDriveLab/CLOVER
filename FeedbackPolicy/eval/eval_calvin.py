import argparse
import glob
import os
import random
from eval_utils import eval_one_epoch_calvin_ddp
from torch.distributed.elastic.multiprocessing.errors import record

os.environ['PYOPENGL_PLATFORM'] = 'egl'
# os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
import numpy as np
import torch
import wandb
from FeedbackPolicy.models.distributed import init_distributed_device, world_info_from_env
from torch.nn.parallel import DistributedDataParallel as DDP

from FeedbackPolicy.models.factory import load_model


def random_seed(seed=42, rank=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


@record
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vision_encoder_path", default="ViT-L-14", type=str)
    parser.add_argument("--vision_encoder_pretrained", default="openai", type=str)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--precision",
        choices=["amp_bf16", "amp_bfloat16", "bf16", "fp16", "fp32"],
        default="fp32",
        help="Floating point precision.",
    )
    parser.add_argument(
        "--calvin_dataset",
        type=str,
        help="path to calvin_dataset",
    )
    parser.add_argument("--calvin_conf_path", type=str, help="path to calvin configuration file")
    parser.add_argument(
        "--visual_planner_checkpoint",
        type=str,
        help="path to checkpoint to evaluate , this should contain model",
        default=None,
    )
    parser.add_argument(
        "--policy_checkpoint",
        type=str,
        help="path to policy checkpoint to evaluate , this should contain model",
        default=None,
    )
    parser.add_argument(
        "--horovod",
        default=False,
        action="store_true",
        help="Use horovod for distributed training.",
    )
    parser.add_argument(
        "--dist-url",
        default="env://",
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument(
        "--no-set-device-rank",
        default=False,
        action="store_true",
        help="Don't set device index from local rank (when CUDA_VISIBLE_DEVICES restricted to one per proc).",
    )
    parser.add_argument(
        "--dist-backend", default="nccl", type=str, help="distributed backend"
    )
    parser.add_argument(
        "--reset",
        default=False,
        action="store_true"
    )
    parser.add_argument(
        "--visualize",
        default=False,
        action="store_true"
    )
    parser.add_argument(
        "--diverse_inst",
        default=False,
        action="store_true"
    )
    parser.add_argument('--sample_step', type=int, default=20, help="diffusion time steps")

    args = parser.parse_args()
    

    args.local_rank, args.rank, args.world_size = world_info_from_env()

    device_id = init_distributed_device(args)
    print("device_id: ", device_id)
    print("world_size: ", torch.distributed.get_world_size())
    random_seed(args.seed)

    diffusion_model, policy_model, tokenizer, text_encoder = load_model(
        args.vision_encoder_path,
        args.vision_encoder_pretrained,
        sample_steps=args.sample_step,
    )

    checkpoint_path = args.visual_planner_checkpoint
    print("Loading checkpoint from ", checkpoint_path)
    diffusion_model.load_state_dict(torch.load(checkpoint_path)['ema'], strict=True)


    diffusion_model = diffusion_model.to(device_id)
    diffusion_model.eval()
    ddp_diffusion_model = DDP(diffusion_model, device_ids=[device_id])

    policy_model = policy_model.to(device_id)
    policy_model.eval()
    ddp_policy_model = DDP(policy_model, device_ids=[device_id])

    checkpoint_path = args.policy_checkpoint
    print("Loading policy checkpoint from ", checkpoint_path)
    ddp_policy_model.load_state_dict(torch.load(checkpoint_path)['model_state_dict'], strict=False)


    ddp_diffusion_model.eval()
    eval_log_dir = None
    if args.visualize:
        eval_log_dir = 'evaluate/{}'.format(args.visual_planner_checkpoint.split('.')[0])
    eval_one_epoch_calvin_ddp(
        args=args,
        model=ddp_diffusion_model,
        policy_model=ddp_policy_model,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        dataset_path=args.calvin_dataset,
        eval_log_dir=eval_log_dir,
        debug=args.visualize,
        reset=args.reset,
        diverse_inst=args.diverse_inst
    )


if __name__ == "__main__":
    os.environ["NCCL_BLOCKING_WAIT"] = '1'
    main()
