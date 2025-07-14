CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch \
    --nproc_per_node=8 --nnodes=1 --node_rank=0 train.py 