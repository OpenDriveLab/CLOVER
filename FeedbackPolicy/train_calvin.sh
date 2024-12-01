#!/bin/bash
export LD_LIBRARY_PATH=/home/pai/envs/openvla/lib/python3.10/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH
# dataset path
calvin_dataset_path='path_to_your/calvin/dataset/task_ABC_D'

subfix=`date "+%Y%m%d-%H%M"`
log_file="logs/training_"${subfix}".log"

torchrun --nnodes=1 --nproc_per_node=8 train/train_calvin.py \
    --vision_encoder vc1-base \
    --num_epochs 10 \
    --gradient_accumulation_steps 1 \
    --batch_size_calvin 16 \
    --run_name feedback_policy_calvin_abc \
    --calvin_dataset ${calvin_dataset_path} \
    --workers 4 \
    --learning_rate 1e-4 \
    --window_size 5 \
    2>&1 | tee ${log_file}
