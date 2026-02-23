#!/bin/sh
# Usage:
#   export CHECKPOINT_ROOT=/path/to/checkpoint/KeyVID
#   bash scripts/generation.sh asva_12_kf
#   bash scripts/generation.sh asva_12_kf_interp
#
# Required environment variables:
#   CHECKPOINT_ROOT     - Checkpoint directory (e.g., ./checkpoint/KeyVID)
#
# Optional environment variables:
#   AVSYNC15_ROOT       - AVSync15 dataset root (default: ./data/AVSync15)
#   DATA_ROOT           - Output directory (default: ./outputs)


EXPSET=$1

# Get the script's directory and project root
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Validate required environment variable
if [ -z "$CHECKPOINT_ROOT" ]; then
    echo "ERROR: CHECKPOINT_ROOT environment variable is not set!"
    echo ""
    echo "Please specify where your checkpoint files are located:"
    echo "  export CHECKPOINT_ROOT=/path/to/checkpoint/KeyVID"
    echo ""
    echo "Example:"
    echo "  export CHECKPOINT_ROOT=$e/checkpoint/KeyVID"
    echo "  bash scripts/generation.sh asva_12_kf"
    echo ""
    echo "Or create a scripts/.env file:"
    echo "  cp scripts/env.example scripts/.env"
    echo "  # Edit scripts/.env to set CHECKPOINT_ROOT"
    echo "  source scripts/.env"
    echo "  bash scripts/generation.sh asva_12_kf"
    exit 1
fi

# Optional paths with defaults
AVSYNC15_ROOT="${AVSYNC15_ROOT:-${PROJECT_ROOT}/data/AVSync15}"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/outputs}"
KEYFRAME_IDX_DIR="${KEYFRAME_IDX_DIR:-${PROJECT_ROOT}/save_results/prediction/motion}"

save_root="${DATA_ROOT}"

# Checkpoint paths
KF_CHECKPOINT="${CHECKPOINT_ROOT}/KeyVID/keyframe_generation/generator_checkpoint.ckpt"
INTERP_CHECKPOINT="${CHECKPOINT_ROOT}/KeyVID/interpolation/epoch=3669-step=58720.ckpt"

# Export data paths for potential use in configs
export AVSYNC15_TRAIN_DIR="${AVSYNC15_ROOT}/train"
export AVSYNC15_TRAIN_KEYFRAME_DIR="${AVSYNC15_ROOT}/train_curves_npy/"
export AVSYNC15_TEST_DIR="${AVSYNC15_ROOT}/test"
export AVSYNC15_TEST_KEYFRAME_DIR="${AVSYNC15_ROOT}/test_curves_npy/"
export AVSYNC15_VIDEOS="${AVSYNC15_ROOT}/videos"
export AVSYNC15_TEST_LIST="${AVSYNC15_ROOT}/test.txt"
export PRETRAINED_CHECKPOINT="${CHECKPOINT_ROOT}/dynamicrafter_512_v1/model.ckpt"


# # 1. Predict keyframes index
# if [ "$EXPSET" == "asva_12_kf_index" ]; then
#     # TODO: Implement keyframe index prediction

keyframe_idx_dir="${KEYFRAME_IDX_DIR}"
if [ "$EXPSET" == "asva_12_kf" ]; then
    keyframe_idx_dir="${KEYFRAME_IDX_DIR}"
elif [ "$EXPSET" == "asva_12_kf_interp" ]; then
    keyframe_gen_dir="outputs/repo/DynamiCrafter/save/asva/asva_12_kf_add_idx_add_fps/epoch=1319-step=15840-kf_audio_7.5_img_2.0_kf_7.5/samples"
    keyframe_idx_dir="${KEYFRAME_IDX_DIR}"
fi

# 2. Generate keyframes
if [ "$EXPSET" == "asva_12_kf" ]; then
    config='configs/inference/keyframe_generation.yaml'
    exp_root=${save_root}'/asva/asva_12_kf_add_idx_add_fps/epoch=1319-step=15840-kf'
    checkpoint="${KF_CHECKPOINT}"
    FS=6
    video_length=12
    INTERP_ARGS="--keyframe_idx_dir $keyframe_idx_dir"

# 3. Interpolate keyframes
elif [ "$EXPSET" == "asva_12_kf_interp" ]; then
    config='configs/inference/keyframe_interpolation.yaml'
    exp_root=${save_root}'/asva/asva_12_kf_interp/reproduce_new_keyframe'
    checkpoint="${INTERP_CHECKPOINT}"
    # Keyframe generation results directory (from Step 1 output)
    FS=24
    video_length=48
    INTERP_ARGS="--interp --keyframe_gen_dir $keyframe_gen_dir --keyframe_idx_dir $keyframe_idx_dir"

else
    echo "Usage: bash scripts/generation.sh [asva_12_kf|asva_12_kf_interp]"
    exit 1
fi

# Validate checkpoint exists
if [ ! -f "$checkpoint" ]; then
    echo "ERROR: Checkpoint file not found: $checkpoint"
    echo "Please ensure the checkpoint is downloaded to the correct location."
    exit 1
fi

# Process config file to replace environment variables
config_processed="${config}.processed"
if [ -f "$config" ]; then
    echo "Processing config: $config"
    envsubst < "$config" > "$config_processed"
    config="$config_processed"
fi

run_asva() {
    local device_id=$1
    local cfg_audio_stage1=$2
    local cfg_img=$3
    local cfg_audio_stage2=$4

    if [ "$EXPSET" == "asva_12_kf" ]; then
        local cfg_audio=${cfg_audio_stage1}
    elif [ "$EXPSET" == "asva_12_kf_interp" ]; then
        local cfg_audio=${cfg_audio_stage2}
    fi

    CUDA_VISIBLE_DEVICES=$device_id python -W ignore scripts/evaluation/animation_gen.py \
        --config ${config} \
        --exp_root ${exp_root}_audio_${cfg_audio_stage1}_img_${cfg_img}_kf_${cfg_audio_stage2} \
        --checkpoint ${checkpoint} \
        --dataset AVSync15 \
        --height 320 \
        --width 512 \
        --video_fps ${FS} \
        --video_length ${video_length} \
        --num_clips_per_video 3 \
        --unconditional_guidance_scale 7.5 \
        --random_seed 0 \
        --ddim_steps 90 \
        --ddim_eta 1.0 \
        --text_input \
        --rank $device_id \
        --timestep_spacing 'uniform_trailing' \
        --guidance_rescale 0.7 \
        --perframe_ae \
        --cfg_audio $cfg_audio \
        --cfg_img $cfg_img \
        --multiple_cond_cfg \
        --keyframe_idx_dir ${KEYFRAME_IDX_DIR} \
        ${INTERP_ARGS}
}

for ((i=0; i<8; i++)); do
    run_asva $i 7.5 2.0 9.0 &
    sleep 1
done

wait
