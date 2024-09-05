
accelerate launch --config_file accelerate_cfg.yaml train.py \
    --val_batch_size 32 \
    --sample_per_seq 8 \
    --sampling_step 5 \
    --results_folder path_to_your_results_folder \
    --resume_path patch_to_your_model_ckpt \
    --with_text_conditioning \
    --diffusion_steps 100 \
    --sample_steps 10 \
    --mode val  \
    --with_depth \
    --flow_reg \
