#!/bin/bash
LOAD='MAGAer13/mplug-owl2-llama2-7b'

DATA_FILE=./IE-R1-train-traindst.json
DUAL_IMAGE_MODE=False

export TOKENIZERS_PARALLELISM=false
output_path=./outputs/base-epoch100-b64


deepspeed --master_port 25801 q_align/train/train_mem.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path $LOAD \
    --version v1 \
    --data_path $DATA_FILE \
    --image_folder ./IE-R1-4K/dst_train/ \
    --dual_image_mode $DUAL_IMAGE_MODE \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ${output_path} \
    --num_train_epochs 100 \
    --per_device_train_batch_size 64 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 2 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --tune_visual_abstractor True \
    --freeze_vision_model False \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to none

cp preprocessor_config.json ${output_path}/
