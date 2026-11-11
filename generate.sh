#!/bin/bash

NUM_GPUS=1
BASE_DIR="${PWD}"

RUNS=(
    "SDV1 SDV1 DDIM 7.5 10 sac_safety_critic 8"
    "MemBench-SDV1 SDV1 DDIM 7.5 10 sac_safety_critic 8"
)

for run in "${RUNS[@]}"; do
    read -r pipe model_name scheduler g N mitigate batch_size <<< "$run"
    
    for seed in $(seq 0 $((N-1))); do
        for train_seed in $(seq 41 45); do
            
            # Format the checkpoint path dynamically based on current variables
            # Note: This looks for a folder named like: checkpoints_SDV1_DDIM_seed_41
            CKPT_DIR="${BASE_DIR}/checkpoints_${model_name}_${scheduler}_seed_${train_seed}"
            
                echo "Generating... (pipe=$pipe, model_name=$model_name, scheduler=$scheduler, g=$g, seed=$seed/$N, train_seed=$train_seed, mitigate=$mitigate) on GPU $gpu_id"
            
            python generate.py \
                --pipe "$pipe" \
                --scheduler "$scheduler" \
                --g "$g" \
                --seed "$seed" \
                --N "$N" \
                --mitigate "$mitigate" \
                --sac-actor-path "${CKPT_DIR}/sac_actor_best.pth" \
                --sac-critic-path "${CKPT_DIR}/sac_critic_best.pth" \
                --sac-encoder-path "${CKPT_DIR}/sac_encoder_best.pth" \
                --sac-safety-critic-path "${CKPT_DIR}/sac_safety_critic_best.pth" \
                --vae-path "${BASE_DIR}/clip_vae/step=45000.pth" \
                --caption-dim 64 \
                --batch-size "$batch_size" \
                --train-seed "$train_seed"
            
            echo ""
        done
    done
done