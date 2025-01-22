
nohup python3 -u main2_10times_degree.py --dataset MUTAG --device 3 --batch_size 32 --epochs 350 --lr 0.01 --num_layers 5 --num_mlp_layers 1 --hidden_dim 32 --final_dropout 0.5 --graph_pooling_type sum --neighbor_pooling_type sum  --learn_eps >>MUTAG.txt





