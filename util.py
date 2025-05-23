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



def entropy_k(adj, k):
    """
    Compute entropy for each node based on k-hop ego network.
    
    Parameters:
        adj (torch.Tensor): Adjacency matrix of shape [n_nodes, n_nodes]
        k (int): Number of hops to include in ego-network (k >= 0)

    Returns:
        torch.Tensor: Entropy score for each node [n_nodes]
    """
    n_nodes = adj.size(0)
    node_entropy = torch.zeros(n_nodes)
    
    for node in range(n_nodes):
       
        neighbors = torch.tensor([node], dtype=torch.long)

        
        frontier = torch.tensor([node], dtype=torch.long)
        visited = set(frontier.tolist())
        
        for _ in range(k):
            new_frontier = []
            for n in frontier:
                neighbors_n = torch.nonzero(adj[n] > 0).squeeze(1)
                for nb in neighbors_n.tolist():
                    if nb not in visited:
                        visited.add(nb)
                        new_frontier.append(nb)
            if not new_frontier:
                break
            frontier = torch.tensor(new_frontier, dtype=torch.long)
            neighbors = torch.unique(torch.cat([neighbors, frontier]))

        subgraph_nodes = neighbors
        subgraph_degrees = adj[subgraph_nodes, :][:, subgraph_nodes].sum(dim=1)
        total_degree = subgraph_degrees.sum()

        p = torch.clamp(subgraph_degrees / total_degree, min=1e-10)
        node_entropy[node] = -(p * p.log()).sum()
    
    return node_entropy


def compute_edge_trussness(graph):
    """
    Compute the trussness of each edge in the graph.
    
    Parameters:
        graph (nx.Graph): Input graph.

    Returns:
        nx.Graph: Graph with trussness added as edge attributes.
    """
    supg = {}

   
    for u, v in graph.edges():
        common_neighbors = set(graph.neighbors(u)).intersection(set(graph.neighbors(v)))
        supg[(u, v)] = len(common_neighbors)
        supg[(v, u)] = len(common_neighbors)

    
    edges_sorted = sorted(graph.edges(), key=lambda e: supg[e])

    k = 2
    GT = graph.copy()

    while edges_sorted:
        for e in edges_sorted:
            u, v = e
            if supg[e] <= k - 2:
                
                common_neighbors = set(GT.neighbors(u)).intersection(set(GT.neighbors(v)))
                for w in common_neighbors:
                    supg[(u, w)] -= 1
                    supg[(w, u)] -= 1
                    supg[(v, w)] -= 1
                    supg[(w, v)] -= 1

                
                GT[u][v]['trussness'] = k - 1
                GT.remove_edge(u, v)

        
        edges_sorted = [e for e in edges_sorted if e in GT.edges()]
        edges_sorted = sorted(edges_sorted, key=lambda e: supg[e])
        if edges_sorted:
            k += 1

    return GT

def TGS(graph, delta=3, eta=3):
    """
    Truss-based graph sparsification (TGS).

    Parameters:
        graph (nx.Graph): Input graph.
        delta (float): Threshold for sparsification.
        eta (int): Minimum trussness cutoff.

    Returns:
        nx.Graph: Sparsified graph.
    """
    
    GT = compute_edge_trussness(graph)


    EH = [(u, v, data['trussness']) for u, v, data in GT.edges(data=True) if data['trussness'] >= eta]
    EH = sorted(EH, key=lambda x: x[2], reverse=True)

    GS = GT.copy()
    for u, v, t_value in EH:
        
        Tu = sum([GT[u][n]['trussness'] for n in GS.neighbors(u)]) / len(set(GS.neighbors(u)))
        Tv = sum([GT[v][n]['trussness'] for n in GS.neighbors(v)]) / len(set(GS.neighbors(v)))
        TN_E = min(Tu, Tv)
        if TN_E >= delta:
            GS.remove_edge(u, v)
            GS = compute_edge_trussness(GS)

    return GS  


def load_data(dataset, degree_as_tag, Truss_process=False, Entropy_process=False, k=1):
    '''
        dataset: name of dataset
        degree_as_tag: the datasets using degree as node label
        Truss_process: Truss-based Process
        Entropy_process: compute the entropy of expansion subgraph
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
            
            if Entropy_process == True:
                adj = get_adj(graph.edge_mat)
                graph.node_entropy = entropy_k(adj, k)
                
            if Truss_process == True:
                Gs = TGS(graph.g)
                graph.g = Gs    

        
        if degree_as_tag:
            for g in g_list:
                g.node_tags = list(dict(g.g.degree(range(len(g.g)))).values())


                
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



        
            
            
                
            
                
        







