torchrun --nproc_per_node=4 \
 --nnodes=1 \
 --node_rank=0 \
 --master_add="192.168.0.97" \
 --master_port=2655 \
 main.py \
 -g 0,1,2,3 \
 -i 0 \
 -n 18 \
 -b 512 \
 -e 50 \
 -d /root/mini-imagenet/train \
 -w 4    