from time import time
import numpy as np

import matplotlib as mpl
mpl.use('Agg')
#...
#...
import matplotlib.pyplot as plt

 
from sklearn import datasets
from sklearn.manifold import TSNE
import torch
from util import load_data, seperate_data
from models.graphcnn2 import GraphCNN
import argparse
from scipy.spatial.distance import jensenshannon


def compute_jsd(row1, row2):
    epsilon = 1e-10
    row1 = row1 + epsilon
    row2 = row2 + epsilon 
    row1 = row1 / np.sum(row1)
    row2 = row2 / np.sum(row2)
    
    return jensenshannon(row1, row2)**2

def cal_JSD(features, graphs):

    start_idx = 0
    average = 0

    for i, graph in enumerate(graphs):
        n = len(graph.g)
        jsd_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:

                    jsd_matrix[i, j] = compute_jsd(features[i+start_idx], features[j+start_idx])

        start_idx = start_idx + n

        jsd_matrix[np.isnan(jsd_matrix)] = 0
        total_sum = np.sum(jsd_matrix)  
        num_elements = n * (n - 1)  
        average += total_sum / num_elements 
    return average/ len(graphs)   

    
 
def main():
    parser = argparse.ArgumentParser(description='PyTorch graph convolution neural net for whole-graph classification')
    parser.add_argument('--dataset', type=str, default="MUTAG", help='name of dataset (default: MUTAG)')
    parser.add_argument('--model', type=str, default="SDE", choices = ("SDE", "ResNet", "DropEdge", "PairNorm", "TGS", "ContraNorm"), help='name of model')
    parser.add_argument('--device', type=int, default=0, help="gpu id of using if any (default: 0)")
    parser.add_argument('--k', type=int, default=1, help="the size of expansion subgraph")
    parser.add_argument('--top_percentage', type=float, default=0.1, help='top percentage for selecting nodes')
    parser.add_argument('--batch_size', type=int, default=32, help="input batch size for training (default: 32)")
    parser.add_argument('--iters_per_epoch', type=int, default=50, help="number of iterations per each epoch (default: 50)")
    parser.add_argument('--epochs', type=int, default=350, help='number of epochs to train (default: 350)')
    parser.add_argument('--lr', type=float, default=0.01, help='leanring rate(default: 0.01)')
    parser.add_argument('--num_layers', type=int, default=5, help='number of layers INCLUDING the input one (default: 5)')
    parser.add_argument('--num_mlp_layers', type=int, default=2, help='number of layers for MLP EXCLUDING the input one (default:2). 1 means linear model.')
    parser.add_argument('--num_views', type=int, default=5, help='number of views (default: 5)')
    parser.add_argument('--hidden_dim', type=int, default=64, help='number of hidden units (default: 64)')
    parser.add_argument('--final_dropout', type=float, default=0.5, help='final layer dropout (default: 0.5)')
    parser.add_argument('--alpha', type=float, default=10, help='loss regulation (default: 0.5)')
    parser.add_argument('--max_iters', type=int, default=100, help='max k-means cluster iterations(default: 100)')
    parser.add_argument('--graph_pooling_type', type=str, default='sum', choices = ['sum', 'average'], help='Pooling for over nodes in a graph: sum or average')
    parser.add_argument('--neighbor_pooling_type', type=str, default="sum", choices=['sum', 'average', 'max'], help='Pooling for over neighboring nodes: sum, average or max')
    parser.add_argument('--learn_eps', action="store_true", help="Whether to learn the epison weighting for the center nodes. Does not affect training accuracy though") 
    parser.add_argument('--degree_as_tag', action="store_true", help='let the input node features be the degree of nodes ')
    

    args = parser.parse_args()
    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device("cuda:"+str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    # print parameters
    for k in args.__dict__:
        print(k +":" +str(args.__dict__[k]))

    graphs, num_classes = load_data(args.dataset, args.degree_as_tag, Truss_process, Entropy_process, args.k)

    ave_train_fold = []
    ave_test_fold = []
    fold_idx = 0
    train_graphs, test_graphs = seperate_data(graphs, args.seed, fold_idx)
    
    
    model = GraphCNN(args.model, args.num_layers, args.num_mlp_layers,  train_graphs[0].node_features.shape[1], args.hidden_dim, num_classes, args.final_dropout, args.learn_eps, args.graph_pooling_type, args.neighbor_pooling_type, device, args.top_percentage).to(device)
    
    model.load_state_dict(torch.load('model_parameters.pth'))
    model.eval()
    features = model(graphs)[1].cpu().detach().numpy()
    print(features.shape)

    jsd_mean = cal_JSD(features, graphs)
    print(jsd_mean)


    
 
 
if __name__ == '__main__':
    main()







    
    






