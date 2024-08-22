#!/bin/bash
export HF_ENDPOINT="https://hf-mirror.com"

MODEL_NAME="stabilityai/stable-diffusion-2-1"
BASE_INSTANCE_DIR="B_dataset"
OUTPUT_DIR_PREFIX="style/style_"
LOSS_DIR_PREFIX="loss/loss_"
RESOLUTION=512
TRAIN_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=1
CHECKPOINTING_STEPS=2000
LEARNING_RATE=1e-5
LR_SCHEDULER="constant"
LR_WARMUP_STEPS=0
MAX_TRAIN_STEPS=40000
SEED=123456
GPU_MIN=0
GPU_MAX=0
FROM=0
TO=0
SAVE_FREQUENCY=4000
DROPOUT=0.1

RANKS=(
    "896" # 03, 09, 10, 16
)

PACKS=(
    "03,09,10,16,11,13,23,24,02,04,22,25,05,08,12,20,00,06,18,27,01,07,15,17,14,19,21,26" # Pack0
)

mkdir loss
mkdir style

for ((folder_number = $FROM; folder_number <= $TO; folder_number+=$GPU_MAX-$GPU_MIN+1)); do
    for ((gpu_id = GPU_MIN; gpu_id <= GPU_MAX; gpu_id++)); do
        current_folder_number=$((folder_number + gpu_id - $GPU_MIN))
        if [ $current_folder_number -gt $MAX_NUM ]; then
            break
        fi
        # INSTANCE_DIR="${BASE_INSTANCE_DIR}/$(printf "%02d" $current_folder_number)/images"
        INSTANCE_DIR="${BASE_INSTANCE_DIR}"
        OUTPUT_DIR="${OUTPUT_DIR_PREFIX}$(printf "%02d" $current_folder_number)"
        LOSS_DIR="${LOSS_DIR_PREFIX}$(printf "%02d" $current_folder_number)"
        CUDA_VISIBLE_DEVICES=$gpu_id
        PROMPT="style_"
        RANK=${RANKS[$folder_number]}
        PACK=${PACKS[$folder_number]}

        COMMAND="CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python train_joint_pack3_3.py \
            --pretrained_model_name_or_path=$MODEL_NAME \
            --instance_data_dir=$INSTANCE_DIR \
            --output_dir=$OUTPUT_DIR \
            --loss_dir=$LOSS_DIR \
            --instance_prompt=$PROMPT \
            --resolution=$RESOLUTION \
            --train_batch_size=$TRAIN_BATCH_SIZE \
            --gradient_accumulation_steps=$GRADIENT_ACCUMULATION_STEPS \
            --learning_rate=$LEARNING_RATE \
            --lr_scheduler=$LR_SCHEDULER \
            --lr_warmup_steps=$LR_WARMUP_STEPS \
            --max_train_steps=$MAX_TRAIN_STEPS \
            --save_frequency=$SAVE_FREQUENCY \
            --dropout=$DROPOUT \
            --rank=$RANK \
            --seed=$SEED \
            --pack=$PACK"

        eval $COMMAND &
        sleep 3
    done
    wait
done