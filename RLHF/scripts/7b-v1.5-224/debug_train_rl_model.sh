#!/bin/bash

set -e
set -x

export CUDA_VISIBLE_DEVICES=4 #0 #,1,2,3
export DATA_DIR="/scratch/e35895/data"
export MODEL_DIR="/scratch/e35895/code/LLaVA-RLHF/checkpoints"
export PYTHONPATH="$PWD:$PYTHONPATH"
export GPUS_PER_NODE=1 #4 #8
export OMP_NUM_THREADS=8
export TRANSFORMERS_OFFLINE=1

# MODEL CONFIG
POLICY_BASE_MODEL_NAME=LLaVA-RLHF-7b-v1.5-224/sft_model
RM_BASE_MODEL_NAME=LLaVA-RLHF-7b-v1.5-224/sft_model

POLICY_LORA=LLaVA-RLHF-7b-v1.5-224/rlhf_lora_adapter_model
RM_LORA=LLaVA-Fact-RM-7b-v1.5-224-sherlock/checkpoint-22000/adapter_model/lora_default  # we use early stopping

# SAVE CONFIG
MODEL_NAME=rl_debug_7b_v1.5_224

# TRAINING CONFIG
LEARNING_RATE=3e-5
KL_COEF=0.1
EPOCH=4
ROLLOUT_BATCH_SIZE=512
STEP_BATCH_SZIE=256
ROLLOUT_PER_DEVICE_BATCH_SIZE=64
REWARD_MODEL_PER_DEVICE_BATCH_SIZE=16
STEP_PER_DEVICE_BATCH_SIZE=16
NOPTEPOCHS=2

# FACT-RLHF CONFIG
INCOMPLETE_RESPONSE=-8.0
LENGTH_BONUS=-10.0
CORRECT_BONUS=2.0

# THINGS CHANGED: RESUME_FROM_TRAINING chnaged to false.

torchrun \
    --standalone \
    --nnodes=1 \
    --nproc-per-node=$GPUS_PER_NODE \
    finetune_lora_ppo.py \
    --do_train \
    --seed 42 \
    --step_batch_size $STEP_BATCH_SZIE \
    --step_per_device_batch_size $STEP_PER_DEVICE_BATCH_SIZE \
    --rollout_batch_size $ROLLOUT_BATCH_SIZE \
    --rollout_per_device_batch_size $ROLLOUT_PER_DEVICE_BATCH_SIZE \
    --reward_model_per_device_batch_size $REWARD_MODEL_PER_DEVICE_BATCH_SIZE \
    --base_model_name "$MODEL_DIR/$POLICY_BASE_MODEL_NAME" \
    --reward_base_model_name "$MODEL_DIR/$RM_BASE_MODEL_NAME" \
    --policy_model_name_or_path "$MODEL_DIR/$POLICY_LORA" \
    --reward_model_name_or_path "$MODEL_DIR/$RM_LORA" \
    --learning_rate $LEARNING_RATE \
    --init_value_with_reward True \
    --warmup_steps 5 \
    --dataset_path $DATA_DIR/LLaVA-RLHF/LLaVA-RLHF-Data/rl_model_debug.json \
    --train_splits "train" \
    --output_dir "$MODEL_DIR/$MODEL_NAME" \
    --total_epochs $EPOCH \
    --group_by_length False \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 10 \
    --save_total_limit 100000 \
    --weight_decay 0.0 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --report_to "tensorboard" \
    --ddp_backend "nccl" \
    --bf16 True \
    --penalty_reward_value $INCOMPLETE_RESPONSE \
    --length_bonus_score $LENGTH_BONUS \
    --correct_bonus_score $CORRECT_BONUS \
    --relative_stop_token_penalty True \
    --penalize_no_stop_token True \
    --ddp_find_unused_parameters False \
    --resume_from_training False \
    --kl_coef $KL_COEF \
    --max_grad_norm 1.0 \
    --whitening_async_stats "full_batch" \
    --clean_tokens_after_eos True \
    --temperature 1.0 \
    --whiten_rewards False \
    --model_max_length 2048 \
    --query_len 128 \
    --response_len 896 \
    --noptepochs $NOPTEPOCHS \
    --image_folder $DATA_DIR/COCO/train2017 \
    --vision_tower different \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --freeze_mm_mlp_adapter True \
    --reward_prompt_file "./prompts/fact_rlhf_reward_prompt.txt" \
    --image_to_caption_file "$DATA_DIR/LLaVA-RLHF/LLaVA-RLHF-Data/image_to_caption.json" \
    --image_aspect_ratio 'pad'
