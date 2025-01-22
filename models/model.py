import torch 
from torch import nn
import torch.nn.functional as F


def mish(x):
    return x *(torch.tanh(F.softplus(x)))

class GraphConvolution(nn.Module):
    def __init__(self, activation =mish):
        super().__init__()
        #self.f_in = f_in
        #self.f_out = f_out
        #self.use_bias = use_bias
        self.activation = activation
        # self.weight = nn.Parameter(torch.FloatTensor(f_in, f_out))
        # self.bias = nn.Parameter(torch.FloatTensor(f_out)) if use_bias else None
        #self.initialize_weights()
    
    # def initialize_weights(self):
    #     if self.activation is None:
    #         nn.init.xavier_uniform_(self.weight)
    #     else:
    #         nn.init.kaiming_uniform_(self.weight, nonlinearity='leaky_relu')
    #     if self.use_bias:
    #         nn.init.zeros_(self.bias)
    def forward(self, input, adj):
        #support = torch.mm(input, self.weight)
        output = torch.mm(adj, input)
        # if self.use_bias:
        #     output.add_(self.bias)
        if self.activation is not None:
            output = self.activation(output)
        return output
    
class GCN(nn.Module):
    def __init__(self, f_in, f_out, use_bias, activation = mish):
        super().__init__()
        self.f_in = f_in
        self.f_out = f_out
        self.use_bias = use_bias
        self.activation = activation
        self.weight = nn.Parameter(torch.FloatTensor(f_in, f_out))
        self.bias = nn.Parameter(torch.FloatTensor(f_out)) if use_bias else None
        self.initialize_weights()
    
    def initialize_weights(self):
        if self.activation is None:
            nn.init.xavier_uniform_(self.weight)
        else:
            nn.init.kaiming_uniform_(self.weight, nonlinearity='leaky_relu')
        if self.use_bias:
            nn.init.zeros_(self.bias)
    def forward(self, input, adj):
        #print(input.shape)
        support = torch.mm(input, self.weight)
        output = torch.mm(adj, support)
        if self.use_bias:
            output.add_(self.bias)
        if self.activation is not None:
            output = self.activation(output)
        return output
    
class GraphCNN(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim, max_iters, final_dropout, graph_pooling_type, device):
        '''
            num_layers: number of layers in the neural networks (INCLUDING the input layer)
            num_mlp_layers: number of layers in mlp (EXCLUDING the input layer)
            input_dim: dimensionality of input features
            hidden_dim: dimensionality of hidden units at ALL layers
            output_dim: number of classes for prediction
            final_output: dropout ratio on the final linear layer
            learn_eps: If true, learn epision to distinguish center nodes from neighbor nodes. If false, aggragate neighbors and center nodes altogether.
            neighbor_pooling_type: how to aggregate neighbors (mean, average, or max)
            graph_pooling_type: how to aggragate entire nodes in a graph (mean, average)
            device : gpu id to use
        
        '''
        super(GraphCNN, self).__init__()

        self.final_dropout = final_dropout
        self.device =device
        self.num_layers = num_layers
        self.num_clusters = hidden_dim
        self.max_iters = max_iters

        self.GCN_layers = torch.nn.ModuleList()
        

        for layer in range(self.num_layers-1):
            if layer == 0:
                self.GCN_layers.append(GraphConvolution())
            else:
                self.GCN_layers.append(GraphConvolution())
        self.lastGCN = GCN(hidden_dim, hidden_dim, use_bias=True)
        
        self.graph_pooling_type = graph_pooling_type


        # self.linears_prediction = torch.nn.ModuleList()
        # #hid_dim = hidden_dim 
        # for layer in range(self.num_layers): #self.num_layers
        #     if layer == 0 or layer == 1:
        #         self.linears_prediction.append(nn.Linear(input_dim, output_dim)) # input layer  0-th iteration  h0
        #     else:
        self.linears_prediction = nn.Linear(hidden_dim, output_dim)
                #hid_dim = hid_dim + hidden_dim  # multi-scale

        
    def __preprocess_graphpool(self, batch_graph):

        start_idx = [0]

        for i, graph in enumerate(batch_graph):
            start_idx.append(start_idx[i] +len(graph.g))
        
        idx = []
        elem = []
        for i, graph in enumerate(batch_graph):
            if self.graph_pooling_type == "average":
                elem.extend([1./len(graph.g)])*len(graph.g)
            else:
                elem.extend([1]*len(graph.g))
            idx.extend([[i, j] for j in range(start_idx[i], start_idx[i+1], 1)])
        elem = torch.FloatTensor(elem)
        idx = torch.LongTensor(idx).transpose(0, 1)
        graph_pool = torch.sparse.FloatTensor(idx, elem, torch.Size([len(batch_graph), start_idx[-1]]))

        return graph_pool.to(self.device)
    
    def CtoD(self, X):
        num_clusters = self.num_clusters
        max_iters=self.max_iters
        
        
        centers = X[:num_clusters, :].clone()
        
        
        for _ in range(max_iters):
            
            distances = torch.cdist(X, centers)
           
            labels = torch.argmin(distances, dim=1)

            for i in range(num_clusters):
                cluster_points = X[labels == i]
                if len(cluster_points) > 0:
                    centers[i] = cluster_points.mean(dim=0)
        
        num_nodes = X.shape[0]
        discrete_X = torch.zeros(num_nodes, num_clusters).to(self.device)
        
        for i in range(X.shape[0]):
            discrete_X[i, labels[i]] = 1
        
        return discrete_X 

    def forward(self, batch_graph):
        X_concat = torch.empty(0).to(self.device)
        #hidden_rep = [torch.empty(0).to(self.device) for _ in range(self.num_layers)]
        for graph in batch_graph:
            X = graph.node_features.to(self.device)
            adj = graph.adj_lap.to(self.device)
            h = X
            #hid_layer = [X]

            for layer in range(self.num_layers-1):
                h = self.CtoD(h).to(self.device)
                h = self.GCN_layers[layer](h, adj)
                #hid_layer.append(h)
            h_graph = self.lastGCN(h, adj)
            X_concat = torch.cat((X_concat, h_graph), dim=0)
            # for i in range(len(hidden_rep)):
            #     hidden_rep[i] = torch.cat((hidden_rep[i], hid_layer[i]), dim=0)
            #print(hidden_rep[i])

        score_over_layer = 0
        graph_pool = self.__preprocess_graphpool(batch_graph)
        pooled_h = torch.spmm(graph_pool, X_concat) # len(batch_graph)* dim of feature
        score_over_layer +=F.dropout(self.linears_prediction(pooled_h), self.final_dropout, training=self.training)
        print(score_over_layer)
        return score_over_layer