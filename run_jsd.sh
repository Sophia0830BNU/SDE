#!/bin/bash

dataset="MUTAG"
model="PairNorm"
log_file="${dataset}_${model}_jsd.txt"

nohup python3 -u cal_JSD.py \
  --dataset $dataset \
  --model $model \
  --device 1 \
  --k 2 \
  --top_percentage 0.1 \
  --batch_size 32 \
  --epochs 20 \
  --lr 0.01 \
  --num_layers 5 \
  --num_mlp_layers 1 \
  --hidden_dim 32 \
  --final_dropout 0.5 \
  --graph_pooling_type sum \
  --neighbor_pooling_type sum \
  --learn_eps \
  >> "$log_file" &
