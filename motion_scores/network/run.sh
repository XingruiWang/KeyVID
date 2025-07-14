CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
    --nproc_per_node=8 --nnodes=1 --node_rank=0 main.py --mode train \