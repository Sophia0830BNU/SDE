import argparse
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from util import load_data, seperate_data
from models.graphcnn import GraphCNN 


criterion = nn.CrossEntropyLoss()

def train(args, model, device, train_graphs, optimizer, epoch):
    model.train()

    total_iters = args.iters_per_epoch
    pbar = range(total_iters)

    loss_accum = 0

    for pos in pbar:
        selected_idx = np.random.permutation(len(train_graphs))[:args.batch_size]
        batch_graph = [train_graphs[idx] for idx in selected_idx]

        output = model(batch_graph)
        labels = torch.LongTensor([graph.label for graph in batch_graph]).to(device)
        loss = criterion(output, labels) 
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss = loss.detach().cpu().numpy()
        loss_accum +=loss

    average_loss = loss_accum/total_iters

    print("epoch: %d"%(epoch))

    print("loss training: %f" %(average_loss))
    return average_loss

def pass_data_iteratively(model, graphs, minibatch_size=64):
    model.eval()
    output = []
    idx = np.arange(len(graphs))
    for i in range(0, len(graphs), minibatch_size):
        sampled_idx = idx[i:i+minibatch_size]
        if len(sampled_idx) ==0:
            continue
        output.append(model([graphs[j] for j in sampled_idx]).detach())
       

    return torch.cat(output, 0)

def test_acc(model, device, graphs):
    output = pass_data_iteratively(model,graphs)
    pred = output.max(1, keepdim=True)[1]
    
    labels = torch.LongTensor([graph.label for graph in graphs]).to(device)
    correct = pred.eq(labels.view_as(pred)).sum().cpu().item()
    acc = correct / float(len(graphs))
    return acc

def test(model, device, train_graphs, test_graphs):
    model.eval()

    acc_train = test_acc(model, device, train_graphs)
    acc_test = test_acc(model, device, test_graphs)

    return acc_train, acc_test

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
    
    if args.model == "TGS":
        Truss_process = True
    else:
        Truss_process = False
        
    if args.model == "SDE":
        Entropy_process = True
    else:
        Entropy_process = False

    graphs, num_classes = load_data(args.dataset, args.degree_as_tag, Truss_process, Entropy_process, args.k)

    Acc_ten_times = []
    for seed in range(10):
        ave_train_fold = []
        ave_test_fold = []
        for fold_idx in range(10):
            train_graphs, test_graphs = seperate_data(graphs, seed, fold_idx)

            
            model = GraphCNN(args.model, args.num_layers, args.num_mlp_layers,  train_graphs[0].node_features.shape[1], args.hidden_dim, num_classes, args.final_dropout, args.learn_eps, args.graph_pooling_type, args.neighbor_pooling_type, device, args.top_percentage).to(device)
            optimizer = optim.Adam(model.parameters(), lr=args.lr)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5) # adjust lr

            acc_train_epochs, acc_test_epochs= [], []

            for epoch in range(1, args.epochs+1):
                scheduler.step()
                ave_loss = train(args, model, device, train_graphs, optimizer, epoch)
                acc_train, acc_test = test(model, device, train_graphs, test_graphs)
                print("Train Acc: {:.4f} \t Test Acc {:.4f} \t".format(acc_train, acc_test))
                acc_train_epochs.append(acc_train)
                acc_test_epochs.append(acc_test)
                #print(model.eps)

            ave_train_fold.append(acc_train_epochs)
            ave_test_fold.append(acc_test_epochs)
            print("Fold: "+str(fold_idx))
            #print("Train Acc: {:.4f} \t Test Acc {:.4f} \t".format(acc_train, acc_test))

        ave_train = np.mean(ave_train_fold, axis=0)
        ave_test = np.mean(ave_test_fold, axis=0)

        max_mean = np.max(ave_test)
        max_index = np.argmax(ave_test)
        print("Best Acc {:.4f} \t Best Epoch {:.4f} \t".format(max_mean, max_index))
        max_mean = max_mean * 100
        Acc_ten_times.append(max_mean)
        
    print("ten times Acc {:.4f} \t std {:.4f} \t".format(np.mean(Acc_ten_times), np.std(Acc_ten_times)))

if __name__ == "__main__":
    main()
