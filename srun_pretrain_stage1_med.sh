#!/bin/bash
GPUS=16
GPUS_PER_NODE=8
PARTITION=MIA_LLM
JOB_NAME=BLIP2pretrain

export TOKENIZERS_PARALLELISM=0
export MASTER_PORT='28500'


srun -n${GPUS} \
    -p ${PARTITION} \
    --job-name=${JOB_NAME} \
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --kill-on-bad-exit=1 \
    python train.py --cfg-path lavis/projects/blip2/train/pretrain_stage1_med.yaml

#    python -m torch.distributed.run --nproc_per_node=$GPUS train.py --cfg-path lavis/projects/blip2/train/pretrain_stage1_med.yaml

#srun -n ${NTASK} \
#  -p ${PARTITION} \
#  --job-name=${JOB_NAME} \
#  --gres=gpu:${GPUS_PER_NODE} \
#  --ntasks-per-node=${NTASK} \
#  --kill-on-bad-exit=1 \
