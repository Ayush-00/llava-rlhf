#!/bin/bash

set -e
set -x

export CUDA_VISIBLE_DEVICES=0 #$(seq -s, 0 $(( $(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l) - 1 )))
export DATA_DIR="/scratch/e35895/data"
export MODEL_DIR="/scratch/e35895/code/LLaVA-RLHF/checkpoints"
export PYTHONPATH="$PWD:$PYTHONPATH"
export GPUS_PER_NODE=1 #$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
export OMP_NUM_THREADS=8
#TORCH_DISTRIBUTED_DEBUG to either INFO or TRACE for more info on errors
# export TORCH_DISTRIBUTED_DEBUG=DETAIL


# MODEL CONFIG
VISION_TOWER=openai/clip-vit-large-patch14
LM_MODEL_NAME=LLaVA-RLHF-7b-v1.5-224/sft_model

# DATA CONFIG
PREFERENCE_DATA=LLaVA-RLHF/sherlock/sherlock_human_pref.json

# SAVE CONFIG
MODEL_NAME=LLaVA-Fact-RM-7b-v1.5-224-sherlock

# TRAINING CONFIG
NUM_EPOCHS=1
LEARNING_RATE=2e-5
BATCH_SIZE=1
GRAD_ACCUMULATION=1

torchrun \
    --standalone \
    --nnodes=1 \
    --nproc-per-node=$GPUS_PER_NODE \
    finetune_lora_rm.py \
    --do_train \
    --do_eval \
    --seed 42 \
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_eval_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUMULATION \
    --model_name_or_path $MODEL_DIR/$LM_MODEL_NAME \
    --image_folder $DATA_DIR/sherlock/images/train \
    --vision_tower $VISION_TOWER \
    --learning_rate $LEARNING_RATE \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --freeze_mm_mlp_adapter True \
    --model_max_length 2048 \
    --query_len 1280 \
    --response_len 768 \
    --dataset_path $DATA_DIR/$PREFERENCE_DATA \
    --eval_dataset_path $DATA_DIR/$PREFERENCE_DATA \
    --dataset_name "none" \
    --eval_dataset_name "none" \
    --eval_size 500 \
    --bits 16 \
    --lora_r 64 \
    --lora_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
    --output_dir "$MODEL_DIR/$MODEL_NAME" \
    --num_train_epochs $NUM_EPOCHS \
    --group_by_length False \
    --evaluation_strategy "steps" \
    --eval_steps 50 \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 2 \
    --weight_decay 0.0 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "constant_with_warmup" \
    --logging_steps 5 \
    --report_to wandb \
    --ddp_backend "nccl" \
    --bf16 True \
    --ddp_find_unused_parameters True \
    --resume_from_training True \
    --reward_prompt_file "./prompts/sherlock_rlhf_reward_prompt.txt" \
    --image_to_caption_file "$DATA_DIR/sherlock/image_to_captions.json" \
    --image_aspect_ratio 'pad'
