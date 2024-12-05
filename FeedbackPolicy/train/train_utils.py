import time
from contextlib import suppress

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm
import itertools
from einops import rearrange


def get_cast_dtype(precision: str):
    cast_dtype = None
    if precision == "bf16" or precision == "amp_bf16":
        cast_dtype = torch.bfloat16
    elif precision == "fp16":
        cast_dtype = torch.float16
    return cast_dtype


def get_autocast(precision):
    if precision == "amp":
        return torch.cuda.amp.autocast
    elif precision == "amp_bfloat16" or precision == "amp_bf16":
        # amp_bfloat16 is more stable than amp float16 for clip training
        return lambda: torch.cuda.amp.autocast(dtype=torch.bfloat16)
    else:
        return suppress
    

def get_ckpt_name(args, epoch=-1):
    if epoch != -1:
        if epoch > 1000:
            ckpt_name += '{}_iter.pth'.format(epoch)
        else:
            ckpt_name += '{}.pth'.format(epoch)
    else:
        ckpt_name += 'final_weights.pth'
    return ckpt_name



def train_one_epoch_calvin(
    args,
    model,
    epoch,
    calvin_loader,
    tokenizer,
    optimizer,
    lr_scheduler,
    device_id,
    wandb,
    window_size
):
    
    num_batches_per_epoch_calvin = calvin_loader.num_batches

    num_batches_per_epoch = num_batches_per_epoch_calvin
    total_training_steps = num_batches_per_epoch * args.num_epochs

    autocast = get_autocast(args.precision)
    cast_dtype = get_cast_dtype(args.precision)

    media_token_id = tokenizer("<image>", add_special_tokens=False)["input_ids"][-1]
    endofchunk_token_id = tokenizer("<|endofchunk|>", add_special_tokens=False)[
        "input_ids"
    ][-1]

    model.train()

    # setup logging
    step_time_m = (
        AverageMeter()
    )  # time for one optimizer step (> 1 batch if using gradient accum)
    data_time_m = (
        AverageMeter()
    )  # avg time to load one batch of both calvin (= 1 batch regardless of gradient accum)
    end = time.time()

    # loop through dataloader
    t = tqdm(
        enumerate(calvin_loader),
        disable=args.rank != 0,
        total=total_training_steps,
        initial=(epoch * num_batches_per_epoch),
    )
    t.set_description(f"epoch {epoch+1}/{args.num_epochs}")
    mv_avg_loss = []
    loss_record = []
    for num_steps, batch_calvin in t:
        data_time_m.update(time.time() - end)
        global_step = num_steps + epoch * num_batches_per_epoch
        
        # put images and labels on device
        images = (batch_calvin[0].to(device_id, dtype=cast_dtype, non_blocking=True))
        labels = batch_calvin[2].to(device_id, dtype=cast_dtype, non_blocking=True)
        gripper = (batch_calvin[3].to(device_id, dtype=cast_dtype, non_blocking=True))
        depth_images = (batch_calvin[-2].to(device_id, dtype=cast_dtype, non_blocking=True))

        # get and clip state tensor into 7-DoFs
        state_tensor = batch_calvin[4].to(device_id, dtype=cast_dtype, non_blocking=True)
        state_tensor = torch.cat([state_tensor[..., :6], state_tensor[..., [-1]]], dim=-1)  
        state_tensor = state_tensor.unsqueeze(2).unsqueeze(2)
        
        labels = [labels[..., :6], (labels[..., 6:] + 1) // 2]


        # run model
        with autocast():
            output = model(
                vision_x=images,
                vision_depth=depth_images,
            )

        ### compute loss
        num_actions, bin_actions = output[0], output[1]#, output[2], output[3]

        velo_label = labels[0]
        grip_label = labels[1]

        loss_calvin_num = 0
        loss_calvin_bin = 0
        # communitive loss over time steps
        for i in range(velo_label.shape[1] - 1):
            loss_calvin_num += torch.nn.functional.huber_loss(num_actions[:, i], velo_label[:, i]) / (velo_label.shape[1] - 1)
            loss_calvin_bin += torch.nn.functional.binary_cross_entropy_with_logits(bin_actions[:, i], grip_label[:, i]) / (velo_label.shape[1] - 1)


        loss_calvin = loss_calvin_num + loss_calvin_bin * 0.1 

        divided_loss_calvin = loss_calvin / args.gradient_accumulation_steps

        #### BACKWARD PASS ####
        loss = (
            divided_loss_calvin * 1.0
        )
        mv_avg_loss.append(loss.item())
        loss_record.append(loss.item())
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # step optimizer and log
        if (((num_steps + 1) % args.gradient_accumulation_steps) == 0) or (
            num_steps == num_batches_per_epoch - 1
        ):
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            # step time and reset end outside of rank 0
            step_time_m.update(time.time() - end)
            end = time.time()

            if args.rank == 0 and args.report_to_wandb:
                # compute within rank 0
                calvin_samples_per_second = (
                    args.gradient_accumulation_steps
                    * args.batch_size_calvin
                    * args.world_size
                    / step_time_m.val
                )
                calvin_samples_per_second_per_gpu = (
                    args.gradient_accumulation_steps
                    * args.batch_size_calvin
                    / step_time_m.val
                )

                wandb.log(
                    {
                        "data_time": data_time_m.avg,
                        "step_time": step_time_m.avg,
                        "calvin_samples_per_second": calvin_samples_per_second,
                        "calvin_samples_per_second_per_gpu": calvin_samples_per_second_per_gpu,
                        "lr": optimizer.param_groups[0]["lr"],
                    },
                    commit=False,
                )
                step_time_m.reset()
                data_time_m.reset()

                wandb.log(
                    {
                        "loss_calvin": divided_loss_calvin.item(),
                        "global_step": global_step,
                        "loss_velo": loss_calvin_num.item(),
                        "loss_grip": loss_calvin_bin.item(),
                    },
                    commit=True,
                )

        
        # Log loss to console
        if ((num_steps + 1) % args.logging_steps == 0) and args.rank == 0:
            print(
                f"Step {num_steps+1}/{num_batches_per_epoch} of epoch {epoch+1}/{args.num_epochs} complete. Loss: (all){loss_calvin.item():.3f} (mse){loss_calvin_num.item():.3f} " + \
                f"(bce){loss_calvin_bin.item():.3f}"
            )
        avg_horizon = min(100, len(mv_avg_loss))
        t.set_postfix({"avg loss": sum(mv_avg_loss[-avg_horizon:]) / avg_horizon, "loss": loss_calvin.item(), "Lnum": loss_calvin_num.item(), "Lbin": loss_calvin_bin.item() })
        

    with open(f'D_loss_log_{args.run_name}.txt', 'a', encoding='utf8') as f:  
        f.write('Average Loss: '+ str(sum(loss_record) / len(loss_record)) + '\n')



def get_checkpoint(model):
    state_dict = model.state_dict()

    for name, p in model.named_parameters():
        if not p.requires_grad and 'normalizer' not in name:
            del state_dict[name]

    return state_dict


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
