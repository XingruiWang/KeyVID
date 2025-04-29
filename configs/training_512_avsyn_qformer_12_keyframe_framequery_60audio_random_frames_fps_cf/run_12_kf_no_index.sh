# NCCL configuration
# export NCCL_DEBUG=INFO
# export NCCL_IB_DISABLE=0
# export NCCL_IB_GID_INDEX=3
# export NCCL_NET_GDR_LEVEL=3
# export NCCL_TOPO_FILE=/tmp/topo.txt

# args
name="training_512_avsyn_qformer_12_keyframe_framequery_60audio_random_frames_fps_cf"
config_file=configs/${name}/config_12_kf_no_index.yaml

# save root dir for logs, checkpoints, tensorboard record, etc.
save_root="save/asva_dynamicrafter"

HOST_GPU_NUM=8



mkdir -p $save_root/$name


python3 -m torch.distributed.launch \
--nproc_per_node=$HOST_GPU_NUM --nnodes=1 --master_addr=127.0.0.1 --master_port=12352 --node_rank=0 \
./main/trainer.py \
--base $config_file \
--train \
--name $name \
--logdir $save_root \
--devices $HOST_GPU_NUM \
lightning.trainer.num_nodes=1
