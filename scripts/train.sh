#!/bin/sh
# Usage:
#   bash scripts/train.sh keyframe_generation
#   bash scripts/train.sh keyframe_interpolation

STAGE=$1
NAME="keyvid_512"

if [ "$STAGE" == "keyframe_generation" ]; then
    CONFIG=configs/training/keyframe_generation.yaml
    SAVE_ROOT="save/asva_12_kf_split_audio_add_frameidx-kf_0_4-adding-fps"
    HOST_GPU_NUM=8
    MASTER_PORT=12352

elif [ "$STAGE" == "keyframe_interpolation" ]; then
    CONFIG=configs/training/keyframe_interpolation.yaml
    SAVE_ROOT="save/asva_12_kf-interp-more"
    HOST_GPU_NUM=8
    MASTER_PORT=12349

else
    echo "Usage: bash scripts/train.sh [keyframe_generation|keyframe_interpolation]"
    exit 1
fi

mkdir -p ${SAVE_ROOT}/${NAME}

python3 -m torch.distributed.launch \
    --nproc_per_node=$HOST_GPU_NUM \
    --nnodes=1 \
    --master_addr=127.0.0.1 \
    --master_port=$MASTER_PORT \
    --node_rank=0 \
    ./main/trainer.py \
    --base $CONFIG \
    --train \
    --name $NAME \
    --logdir $SAVE_ROOT \
    --devices $HOST_GPU_NUM \
    lightning.trainer.num_nodes=1
