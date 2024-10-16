#!/bin/bash
export EVALUTION_ROOT=$(pwd)

# Set CALVIN path
calvin_dataset_path='path_to_your/calvin/dataset/task_ABC_D'
calvin_conf_path="path_to_your/calvin/calvin_models/conf"

# Set checkpoints path
visual_planner_checkpoint='path_to_your/visual_planner.pt'
policy_checkpoint='path_to_your/feedback_policy.pth'

export MESA_GL_VERSION_OVERRIDE=4.1
node_num=4


torchrun --nnodes=1 --nproc_per_node=${node_num} --master_port=6600 eval/eval_calvin.py \
    --visual_planner_checkpoint ${visual_planner_checkpoint} \
    --policy_checkpoint ${policy_checkpoint} \
    --calvin_dataset ${calvin_dataset_path} \
    --calvin_conf_path ${calvin_conf_path} \
    --sample_step 20 \


