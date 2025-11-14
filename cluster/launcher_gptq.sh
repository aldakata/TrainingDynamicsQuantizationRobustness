#!/bin/bash
SUB_FILE=/home/atatjer/src/TrainingDynamicsQuantizationRobustness/cluster/quantize.sub
CONFIG_DIR=/home/atatjer/src/TrainingDynamicsQuantizationRobustness/config/gptq

experiments=(
    "quantize_olmo2_checkpoints.yaml 4" 
)

for exp in "${experiments[@]}"; do
    read -r cfg nrun <<< "$exp"
    full_path="${CONFIG_DIR}/${cfg}"
    cfg_name="${cfg%.*}"

    echo "Submitting: $cfg_name with nrun=$nrun"
    condor_submit_bid 50 "$SUB_FILE" \
        -append "config=$full_path" \
        -append "queue $nrun"
done