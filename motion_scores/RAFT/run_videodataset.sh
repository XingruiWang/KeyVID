

# VGG_PATH=/dockerx/local/data/VGGSound/scratch/shared/beegfs/hchen/train_data/VGGSound_final/video
# debug_PATH=/dockerx/local/repo/DynamiCrafter/data/AVSync15/test_frames/
# SAVE_PATH=/dockerx/local/data/VGGSound_audio_scores


VIDEO_PATH=/datapath/AVSync15/train
SAVE_PATH=/datapath/AVSync15/train_npy


RANK=$1
NODE=$2

for RANK in 0 1 2 3 4 5 6 7
do
    CUDA_VISIBLE_DEVICES=$RANK python demo.py \
        --dataset_root $VIDEO_PATH \
        --output_path $SAVE_PATH \
        --mixed_precision \
        --rank $RANK \
        --node 0 \
        --N 8 \
        --batch_size 128 \
        --all_instances all_instances.txt &
done
