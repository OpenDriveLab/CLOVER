""" Main training script """

import argparse
import copy
import glob
import os
import random
from collections import OrderedDict
import numpy as np
import torch
import wandb
from huggingface_hub import hf_hub_download

from torch.nn.parallel import DistributedDataParallel as DDP

from FeedbackPolicy.data.data import get_data
from FeedbackPolicy.train.distributed import init_distributed_device, world_info_from_env
from train_utils import  train_one_epoch_calvin, get_ckpt_name
from torch.distributed.elastic.multiprocessing.errors import record
from transformers import (
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)

from FeedbackPolicy.models.factory import create_feedback_policy


def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)

def adjust_learning_rate(optimizer, epoch):
    lr = optimizer.param_groups[0]['lr'] * 0.1  
    return lr


@record
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vision_encoder", default="vc1-base", type=str)
    parser.add_argument(
        "--run_name",
        type=str,
        default="RobotFlamingo",
        help="used to name saving directory and wandb run",
    )
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--window_size", type=int, default=5)
    parser.add_argument("--sampling_step", type=int, default=1)
    parser.add_argument(
        "--logging_steps", type=int, default=100, help="log loss every n steps"
    )
    # Sum of gradient optimization batch size
    parser.add_argument("--batch_size_calvin", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        help="path to checkpoint to resume from, this should contain model, optimizer, and lr_scheduler states",
        default="feedback_policy.pth",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning_rate", default=1e-4, type=float)  # 1e-4
    parser.add_argument(
        "--calvin_dataset",
        type=str,
        help="path to calvin_dataset",
    )
    parser.add_argument("--warmup_steps", default=5000, type=int)
    parser.add_argument("--local-rank", default=0, type=int)
    parser.add_argument("--weight_decay", default=0.1, type=float)
    parser.add_argument(
        "--precision",
        choices=["amp_bf16", "amp_bfloat16", "bf16", "fp16", "fp32"],
        default="fp32",
        help="Floating point precision.",
    )
    # data args
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--train_num_samples_calvin", type=int, default=100)
    parser.add_argument("--dataset_resampled", action="store_true")
    # distributed training args
    parser.add_argument(
        "--dist-url",
        default="env://",
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument(
        "--dist-backend", default="nccl", type=str, help="distributed backend"
    )
    parser.add_argument(
        "--horovod",
        default=False,
        action="store_true",
        help="Use horovod for distributed training.",
    )
    parser.add_argument(
        "--no-set-device-rank",
        default=False,
        action="store_true",
        help="Don't set device index from local rank (when CUDA_VISIBLE_DEVICES restricted to one per proc).",
    )
    # wandb args
    parser.add_argument("--report_to_wandb", default=False, action="store_true")
    parser.add_argument(
        "--wandb_project",
        type=str,
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
    )
    
    args = parser.parse_args()

    args.local_rank, args.rank, args.world_size = world_info_from_env()

    device_id = init_distributed_device(args)
    print("device_id: ", device_id)

    
    # Prepare models
    model, image_processor, tokenizer = create_feedback_policy(
        args.vision_encoder,
        args.resume_from_checkpoint,
    )


    calvin_dataset = get_data(args, image_processor, tokenizer, "calvin")
    random_seed(args.seed, args.rank)

    print(f"Start running training on rank {args.rank}.")

    if args.rank == 0 and args.report_to_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.run_name,
            config=vars(args),
        )

    device_id = args.rank % torch.cuda.device_count()
    if args.precision == "bf16" or args.precision == "amp_bfloat16" or args.precision == "amp_bf16":
        model = model.bfloat16()
    elif args.precision == "fp16":
        model = model.half()
    else:
        model = model.float()


    model = model.to(device_id)
    ddp_model = DDP(model, device_ids=[device_id], find_unused_parameters=False)

    args.learning_rate = args.learning_rate 
    optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=args.learning_rate)

    total_training_steps = calvin_dataset.dataloader.num_batches * args.num_epochs

    if args.rank == 0:
        print(f"Total training steps: {total_training_steps}")

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = int(total_training_steps * 0.7), gamma=0.1)


    for epoch in range(args.num_epochs):
        calvin_dataset.set_epoch(epoch)
        calvin_loader = calvin_dataset.dataloader

        train_one_epoch_calvin(
            args=args,
            model=ddp_model,
            epoch=epoch,
            tokenizer=tokenizer,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            calvin_loader=calvin_loader,
            device_id=device_id,
            wandb=wandb,
            window_size = args.window_size
        )

        if args.rank == 0:
            # pass
            if not os.path.exists(args.run_name):
                os.makedirs(args.run_name)

            checkpoint_dict = {
                "epoch": epoch,
                "model_state_dict": get_checkpoint(ddp_model),
                "optimizer_state_dict": optimizer.state_dict(),
                "lr_scheduler_state_dict": lr_scheduler.state_dict(),
            }

            ckpt_name = get_ckpt_name(args, epoch)
            ckpt_path = os.path.join(args.run_name, ckpt_name)

            print(f"Saving checkpoint to {ckpt_path}")
            torch.save(checkpoint_dict, ckpt_path)
            if args.delete_previous_checkpoint:
                if epoch > 0:
                    os.remove(ckpt_path)

    if args.rank == 0:
        if not os.path.exists(args.run_name):
            os.makedirs(args.run_name)

        ckpt_name = get_ckpt_name(args,)
        torch.save(get_checkpoint(ddp_model), f"{args.run_name}/{ckpt_name}")
        if args.report_to_wandb and args.save_checkpoints_to_wandb:
            wandb.save(f"{args.run_name}/{ckpt_name}")


if __name__ == "__main__":
    main()
