import networkx as nx
import numpy as np
import torch 
import random
from scipy.sparse import coo_matrix, diags, eye
from sklearn.model_selection import StratifiedKFold

class S2VGraph(object):
    def __init__(self, g, label, node_tags=None, node_features=None):
        '''
            g: a networkx graph
            label: an integer raph label
            node_tags: a list of integer node tags
            node_features: a torch float tensor, one-hot representation of the tag that is used as input to nerual nets
            edge_mat: a torch long tensor, contain edge list, will be used to create torch sparse tensor
            neighbors: list of neighbors (without self-loop)
        '''
        self.label = label
        self.g = g
        self.node_tags = node_tags
        self.neighbors = []
        self.node_features = 0
        self.edge_mat = 0
        self.adj_lap = 0

        self.max_neighbor = 0
        self.degree_list = 0
        
        self.node_entropy = 0 
        self.adj_js = 0
        
        self.row_sum = 0
        
        

def get_adj(edge_mat):
    num_nodes = edge_mat.max().item() + 1
    adj = torch.zeros(num_nodes, num_nodes, dtype=torch.float32)

    for i in range(edge_mat.size(1)):
        src = edge_mat[0, i]
        dst = edge_mat[1, i]
        adj[src, dst] = 1
        adj[dst, src] = 1 
    return adj 

def normalize(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = (1/rowsum).flatten()
    r_inv[np.isinf(r_inv)] =0
    r_mar_inv = diags(r_inv)
    mx = r_mar_inv.dot(mx)
    return mx

### Laplacian maitrix
def normalize_adj(mx):
    rowsum = mx.sum(dim=1)
    r_inv_sqrt = torch.pow(rowsum, -0.5)
    r_inv_sqrt[torch.isinf(r_inv_sqrt)] = 0
    r_mat_inv_sqrt = torch.diag(r_inv_sqrt)
    mx_normalized = mx.matmul(r_mat_inv_sqrt).transpose(0, 1).matmul(r_mat_inv_sqrt)
    return mx_normalized

def js_kernel(V_p, V_q, H_p, H_q, lambda_param = 1.0):
    
    weight_p = (2 * V_p - V_q) / (2 * V_p + 2 * V_q)
    weight_q = (2 * V_q - V_p) / (2 * V_p + 2 * V_q)
    weight_entropy = weight_p * H_p + weight_q * H_q
    
    k_js = torch.exp(-lambda_param*weight_entropy)
    
    return k_js
    

def entropy(adj):
    
    n_nodes = adj.size(0)
    node_entropy = torch.zeros(n_nodes)
    node_numbers = torch.zeros(n_nodes)
    #node_p = torch.zeros(n_nodes)
    
    for node in range(n_nodes):
        neighbors_1st = torch.nonzero(adj[node]>0).squeeze(1)
        
        neighbors_2nd = torch.tensor([], dtype=torch.long)
        # print(torch.tensor([node]))
        for neighbor in neighbors_1st:
            neighbors_2nd = torch.cat([neighbors_2nd, torch.nonzero(adj[neighbor] > 0).squeeze(1)])
            
        neighbors = torch.unique(torch.cat([torch.tensor([node]), neighbors_1st, neighbors_2nd]))
        #subgraph_nodes = torch.cat((torch.tensor([node]), neighbors), dim=0) 

        node_numbers[node] = len(neighbors)
        subgraph_nodes = neighbors
        subgraph_degrees = adj[subgraph_nodes,:][:, subgraph_nodes].sum(dim=1)
        total_degree = subgraph_degrees.sum()
        
        p = torch.clamp(subgraph_degrees/total_degree, min=1e-10)
        entropy = -(p*p.log()).sum()
        
        #node_p[node] = p
        node_entropy[node] = entropy
        
    
    
    # adj_js = torch.zeros((n_nodes,n_nodes))
    
    # for i in range(n_nodes):
    #     for j in range(i, n_nodes):
    #         if i == j:
    #             adj_js[i, j] = 0
    #         else:
    #             adj_js[i, j] = js_kernel(node_numbers[i], node_numbers[j], node_entropy[i], node_entropy[j])
    #             adj_js[j, i] = adj_js[i, j]
                
    # adj_sums = adj_js.sum(dim=1, keepdim=True)
    # norm_adj_js = adj_js / (adj_sums + 1e-10)

        
    return node_entropy 



def entropy_1(adj):
    n_nodes = adj.size(0)
    
    # 只计算一阶邻居
    neighbors_1st = [torch.nonzero(adj[node] > 0).squeeze(1) for node in range(n_nodes)]
    
    node_entropy = torch.zeros(n_nodes)
    node_numbers = torch.zeros(n_nodes)
    
    for node in range(n_nodes):
        # 只考虑节点和它的一阶邻居
        neighbors = torch.unique(torch.cat([torch.tensor([node]), neighbors_1st[node]]))
        
        node_numbers[node] = len(neighbors)
        
        # 计算子图的度数
        subgraph_degrees = adj[neighbors, :][:, neighbors].sum(dim=1)
        total_degree = subgraph_degrees.sum()
        
        # 计算概率和熵
        p = torch.clamp(subgraph_degrees / total_degree, min=1e-10)
        entropy = -(p * p.log()).sum()
        
        node_entropy[node] = entropy
    
    return node_entropy 

def entropy_2(adj):
    n_nodes = adj.size(0)
    node_entropy = torch.zeros(n_nodes)
    node_numbers = torch.zeros(n_nodes)
    #node_p = torch.zeros(n_nodes)
    
    for node in range(n_nodes):
        neighbors_1st = torch.nonzero(adj[node]>0).squeeze(1)
        
        neighbors_2nd = torch.tensor([], dtype=torch.long)
        # print(torch.tensor([node]))
        for neighbor in neighbors_1st:
            neighbors_2nd = torch.cat([neighbors_2nd, torch.nonzero(adj[neighbor] > 0).squeeze(1)])
            
        neighbors = torch.unique(torch.cat([torch.tensor([node]), neighbors_1st, neighbors_2nd]))
        #subgraph_nodes = torch.cat((torch.tensor([node]), neighbors), dim=0) 

        node_numbers[node] = len(neighbors)
        subgraph_nodes = neighbors
        subgraph_degrees = adj[subgraph_nodes,:][:, subgraph_nodes].sum(dim=1)
        total_degree = subgraph_degrees.sum()
        
        p = torch.clamp(subgraph_degrees/total_degree, min=1e-10)
        entropy = -(p*p.log()).sum()
        
        #node_p[node] = p
        node_entropy[node] = entropy
        
    
    
    # adj_js = torch.zeros((n_nodes,n_nodes))
    
    # for i in range(n_nodes):
    #     for j in range(i, n_nodes):
            
    #         if node_numbers[i] == 1 or node_numbers[j] == 1:
    #             adj_js[i, j] = adj_js[j, i] = 0
    #         else:
    #             adj_js[i, j] = js_kernel(node_numbers[i], node_numbers[j], node_entropy[i], node_entropy[j])
    #             adj_js[j, i] = adj_js[i, j]
    #         # if torch.isnan(adj_js[i, j]):
    #         #     print(node_numbers[i], node_numbers[j], node_entropy[i], node_entropy[j])
    
    # diag = torch.diag(adj_js) 
    
    # # if True in torch.isnan(adj_js):
    # #     print(adj_js)
    
    # epsilon = 1e-12
    # diag_sqrt = torch.sqrt(diag + epsilon)        
    # normalize_factor = diag_sqrt.unsqueeze(0) * diag_sqrt.unsqueeze(1)
    
    # norm_adj_js = adj_js / normalize_factor
    
    # if True in torch.isnan(norm_adj_js):
    #     print(norm_adj_js)
    
    
    
        
    return node_entropy   


def entropy_k3(adj):
    n_nodes = adj.size(0)
    node_entropy = torch.zeros(n_nodes)
    node_numbers = torch.zeros(n_nodes)
    #node_p = torch.zeros(n_nodes)
    
    for node in range(n_nodes):
        neighbors_1st = torch.nonzero(adj[node]>0).squeeze(1)
        
        neighbors_2nd = torch.tensor([], dtype=torch.long)
        # print(torch.tensor([node]))
        for neighbor in neighbors_1st:
            neighbors_2nd = torch.cat([neighbors_2nd, torch.nonzero(adj[neighbor] > 0).squeeze(1)])
            
        
        
        neighbors_3rd = torch.tensor([], dtype=torch.long)
        for neighbor in neighbors_2nd:
            neighbors_3rd = torch.cat([neighbors_3rd, torch.nonzero(adj[neighbor] > 0).squeeze(1)])
            
        neighbors = torch.unique(torch.cat([torch.tensor([node]), neighbors_1st, neighbors_2nd]))
        #subgraph_nodes = torch.cat((torch.tensor([node]), neighbors), dim=0) 

        node_numbers[node] = len(neighbors)
        subgraph_nodes = neighbors
        subgraph_degrees = adj[subgraph_nodes,:][:, subgraph_nodes].sum(dim=1)
        total_degree = subgraph_degrees.sum()
        
        p = torch.clamp(subgraph_degrees/total_degree, min=1e-10)
        entropy = -(p*p.log()).sum()
        
        #node_p[node] = p
        node_entropy[node] = entropy
        
    
    
    # adj_js = torch.zeros((n_nodes,n_nodes))
    
    # for i in range(n_nodes):
    #     for j in range(i, n_nodes):
            
    #         if node_numbers[i] == 1 or node_numbers[j] == 1:
    #             adj_js[i, j] = adj_js[j, i] = 0
    #         else:
    #             adj_js[i, j] = js_kernel(node_numbers[i], node_numbers[j], node_entropy[i], node_entropy[j])
    #             adj_js[j, i] = adj_js[i, j]
    #         # if torch.isnan(adj_js[i, j]):
    #         #     print(node_numbers[i], node_numbers[j], node_entropy[i], node_entropy[j])
    
    # diag = torch.diag(adj_js) 
    
    # # if True in torch.isnan(adj_js):
    # #     print(adj_js)
    
    # epsilon = 1e-12
    # diag_sqrt = torch.sqrt(diag + epsilon)        
    # normalize_factor = diag_sqrt.unsqueeze(0) * diag_sqrt.unsqueeze(1)
    
    # norm_adj_js = adj_js / normalize_factor
    
    # if True in torch.isnan(norm_adj_js):
    #     print(norm_adj_js)
    
    
    
        
    return node_entropy     
        



def get_adj_entropy(adj, node_entropy):
    
    n = adj.size(0)
    adj_e = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i+1, n):
            diff_entropy = abs(node_entropy[i] - node_entropy[j]) / min(node_entropy[i], node_entropy[j])
            adj_e[i, j] = diff_entropy
            adj_e[j, i] = diff_entropy
    
    return adj_e


def load_data(dataset, degree_as_tag):
    '''
        dataset: name of dataset
        test_proportion: ratio of test train split
        seed: random seed for random splitting of dataset
    '''

    print("loading data")
    g_list = []
    label_dict ={}
    feat_dict = {}

    num_nodes = 0 # total number of nodes

    with open('dataset/dataset/%s/%s.txt' %(dataset, dataset), 'r') as f:
        n_g = int(f.readline().strip()) # number of graph
        for i in range(n_g):
            row = f.readline().strip().split()
            n, l = [int(w) for w in row] # n: number of nodes, l: graph label
            num_nodes += n
            if not l in label_dict:
                mapped = len(label_dict)
                label_dict[l] = mapped
            g = nx.Graph()
            node_tags = []
            node_features = []
            n_edges = 0
            for j in range(n):
                g.add_node(j)
                row = f.readline().strip().split()
                tmp = int(row[1])+2
                if tmp == len(row):
                    # no nodes attributes
                    row = [int(w) for w in row]
                    attr = None
                else:
                    row, attr = [int(w) for w in row[:tmp]], np.array([float(w) for w in row[tmp:]])
                if not row[0] in feat_dict:
                    mapped = len(feat_dict)
                    feat_dict[row[0]] = mapped
                node_tags.append(feat_dict[row[0]])

                if tmp > len(row):
                    node_features.append(attr)
                n_edges += row[1]
                for k in range(2, len(row)):
                    g.add_edge(j, row[k])
            if node_features !=[]:
                node_features = np.stack(node_features)
                node_feature_flag = True
            else:
                node_features  = None
                node_feature_flag = False
            
            assert len(g) == n

            g_list.append(S2VGraph(g, l, node_tags))
        
        for graph in g_list:
            # print(graph.adj)
            graph.neighbors = [[] for i in range(len(graph.g))]
            for i, j in graph.g.edges():
                graph.neighbors[i].append(j)
                graph.neighbors[j].append(i)
            degree_list = []
            for i in range(len(graph.g)):
                degree_list.append(len(graph.neighbors[i]))
            graph.degree_list = degree_list
            graph.max_neighbor = max(degree_list)

            graph.label = label_dict[graph.label]

            edges = [list(pair) for pair in graph.g.edges()]
            
            edges.extend([[i, j] for j, i in edges])
            graph.edge_mat = torch.LongTensor(edges).transpose(0, 1)
            adj = get_adj(graph.edge_mat)
            #graph.node_entropy, graph.adj_js = entropy_k3(adj)
            graph.node_entropy = entropy_1(adj)
            # graph.row_sum = graph.adj_js.sum(dim=1)
            # #print(graph.row_sum.shape)
            
            # graph.entropy_adj = get_adj_entropy(adj,graph.node_entropy)
            # self_loop = torch.diag(torch.ones_like(adj[:, 0]))
            # adj_loop = adj + self_loop
            # adj_lap = normalize_adj(adj_loop)
            # graph.adj_lap = adj_lap
        if degree_as_tag:
            for g in g_list:
                g.node_tags = list(dict(g.g.degree(range(len(g.g)))).values())
                #print(g.node_tags)

        # Extracting unique tag labels 
                
        tagset = set([])
        for g in g_list:
            tagset = tagset.union(set(g.node_tags))
        
        tagset = list(tagset)
        tag2index = {tagset[i]: i for i in range(len(tagset))}

        # one hot encoding
        for g in g_list:
            g.node_features = torch.zeros(len(g.node_tags), len(tagset))
            g.node_features[range(len(g.node_tags)), [tag2index[tag] for tag in g.node_tags]] = 1

        print("# classes: %d" % len(label_dict))
        print("# maximum node tag: %d"% len(tagset))
        print("# data :%d"%len(g_list))

        print("# total number: %d"% num_nodes)
        
        return g_list, len(label_dict)
    
def seperate_data(graph_list, seed, fold_idx):
    assert 0<= fold_idx and fold_idx < 10, "fold_idx must be from 0 to 9."
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state =seed)

    labels = [graph.label for graph in graph_list]

    idx_list = []
    for idx in skf.split(np.zeros(len(labels)), labels):
        idx_list.append(idx)

    train_idx, test_idx = idx_list[fold_idx]
    train_graph_list = [graph_list[i] for i in train_idx]
    test_graph_list = [graph_list[i] for i in test_idx]

    return train_graph_list, test_graph_list



        
            
            
                
            
                
        







