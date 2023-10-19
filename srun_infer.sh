#!/bin/bash
GPUS=1
GPUS_PER_NODE=1
PARTITION=MIA_LLM
JOB_NAME=blip_instruct_infer

export TOKENIZERS_PARALLELISM=0
export MASTER_PORT='28500'


srun -n${GPUS} \
    -p ${PARTITION} \
    --job-name=${JOB_NAME} \
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --kill-on-bad-exit=1 \
    python infer.py

