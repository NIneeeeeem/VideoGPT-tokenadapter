#!/bin/sh

GPUS=$1
export DATASET_DIR=.cache/instruction_data

# 设置PYTHONPATH环境变量,change here
export PYTHONPATH="/hhd2/wxc/VideoGPT-plus:$PYTHONPATH"

# qwen weight的位置，visual encoder的位置
BASE_LLM_PATH=.cache/qwen2_5/1.5b_instruction
VISION_TOWER=.cache/InternVideo2-Stage2_1B-224p-f4
IMAGE_VISION_TOWER=.cache/clip-vit-large-patch14-336
PROJECTOR_TYPE=mlp2x_gelu
# 预训练的porjector权重，这部分权重待上传huggingface，其中:
# PRETRAIN_IMAGE_MLP_PATH 是 image_mm_projector.bin
# PRETRAIN_VIDEO_MLP_PATH 是 mm_projector.bin
PRETRAIN_VIDEO_MLP_PATH=results/pretrain/mlp2x_gelu_internvideo2/mm_projector.bin
PRETRAIN_IMAGE_MLP_PATH=results/pretrain/mlp2x_gelu_clip_l14_336px/mm_projector.bin
# 输出权重位置，用于后续测试
OUTPUT_DIR_PATH=results/pool_full/finetune_qwen1b_poolbranch_pool2

# pool_level 用于调节 pool 层，进而调整 visual token 数量
deepspeed --include localhost:${GPUS} --master_port 35381 videogpt_plus/train/train.py \
--lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
--deepspeed scripts/zero3.json \
--model_name_or_path "$BASE_LLM_PATH" \
--version conv_qwen2_cap \
--pool_level 2 \
--dataset_use MVBench_FINETUNING \
--vision_tower "$VISION_TOWER" \
--image_vision_tower "$IMAGE_VISION_TOWER" \
--mm_projector_type "$PROJECTOR_TYPE" \
--image_mm_projector_type "$PROJECTOR_TYPE" \
--pretrain_mm_mlp_adapter "$PRETRAIN_VIDEO_MLP_PATH" \
--pretrain_image_mm_mlp_adapter "$PRETRAIN_IMAGE_MLP_PATH" \
--mm_vision_select_layer -2 \
--mm_use_im_start_end False \
--mm_use_im_patch_token False \
--image_aspect_ratio pad \
--group_by_modality_length True \
--bf16 True \
--output_dir $OUTPUT_DIR_PATH \
--num_train_epochs 1 \
--per_device_train_batch_size 4 \
--per_device_eval_batch_size 4 \
--gradient_accumulation_steps 8 \
--evaluation_strategy "no" \
--save_strategy "steps" \
--save_steps 50000 \
--save_total_limit 1 \
--learning_rate 2e-4 \
--weight_decay 0. \
--warmup_ratio 0.03 \
--lr_scheduler_type "cosine" \
--logging_steps 1 \
--tf32 True \
--model_max_length 4096 \
--gradient_checkpointing True \
--dataloader_num_workers 8 \
--lazy_preprocess True \
--report_to none
