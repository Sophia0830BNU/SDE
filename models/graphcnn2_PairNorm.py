import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append("models/")
from mlp import MLP


# gin discrete

class GraphCNN(nn.Module):
    def __init__(self, num_layers, num_mlp_layers, input_dim, hidden_dim, output_dim, final_dropout, learn_eps, graph_pooling_type, neighbor_pooling_type, device):
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
        self.graph_pooling_type = graph_pooling_type
        self.neighbor_pooling_type = neighbor_pooling_type
        self.learn_eps = learn_eps
        #self.eps0 = nn.Parameter(torch.zeros(self.num_layers-1))
        #self.eps1 = nn.Parameter(torch.zeros(self.num_layers-1))
        self.eps2 = nn.Parameter(torch.zeros(self.num_layers-1))

        # List of MLPS
        #self.mlps0 = torch.nn.ModuleList()
        #self.mlps1 = torch.nn.ModuleList()
        self.mlps2 = torch.nn.ModuleList()

        # List of batchnorms applied to the output of MLP
        #self.batch_norms0 = torch.nn.ModuleList()
        #self.batch_norms1 = torch.nn.ModuleList()
        self.batch_norms2 = torch.nn.ModuleList()
        
        #self.assign = torch.nn.ModuleList()

        # hidden_d = []

        # for layer in range(self.num_layers):
        #     hidden_d.append(hidden_dim)
        #     hidden_dim = hidden_dim *2

        for layer in range(self.num_layers-1):
            if layer == 0 :
                #self.mlps0.append(MLP(num_mlp_layers, input_dim, hidden_dim, hidden_dim))
                #self.mlps1.append(MLP(num_mlp_layers, input_dim, hidden_d[layer], hidden_d[layer]))
                self.mlps2.append(MLP(num_mlp_layers, input_dim, hidden_dim, hidden_dim))
            else:
                #self.mlps0.append(MLP(num_mlp_layers, hidden_dim, hidden_dim, hidden_dim))
                #self.mlps1.append(MLP(num_mlp_layers, hidden_d[layer-1], hidden_d[layer], hidden_d[layer]))
                self.mlps2.append(MLP(num_mlp_layers, hidden_dim, hidden_dim, hidden_dim))
            #self.batch_norms0.append(nn.BatchNorm1d(hidden_dim))
            #self.batch_norms1.append(nn.BatchNorm1d(hidden_d[layer]))
            self.batch_norms2.append(nn.BatchNorm1d(hidden_dim))
            # self.assign.append(nn.Linear(hidden_dim, hidden_dim))
        
        
        # self.linear_softmax = torch.nn.ModuleList()
        # for layer in range(self.num_layers-2):
        #     self.linear_softmax.append(nn.Linear(hidden_d[layer*2], hidden_d[layer*2+1]))

        # self.attention = torch.nn.ModuleList()
        # for layer in range(self.num_layers - 1):
        #     self.attention.append(Attention(hidden_d[layer]))

        self.linears_prediction = torch.nn.ModuleList()
        #hid_dim = hidden_dim 
        for layer in range(self.num_layers): #self.num_layers
            if layer == 0:
                self.linears_prediction.append(nn.Linear(input_dim, output_dim)) # input layer  0-th iteration  h0
            else:
                self.linears_prediction.append(nn.Linear(hidden_dim, output_dim))
                #hid_dim = hid_dim + hidden_dim  # multi-scale



    def __preprocess_neighbors_maxpool(self, batch_graph):

        max_deg = max([graph.max_neighbor for graph in batch_graph])
        #print(max_deg)
        padded_neighbor_list = []
        start_idx = [0]

        for i, graph in enumerate(batch_graph):
            start_idx.append(start_idx[i] +len(graph.g))
            padded_neighbors = []
            for j in range(len(graph.neighbors)):
                pad = [n + start_idx[i] for n in graph.neighbors[j]]
                pad.extend([-1]*(max_deg - len(pad)))

                if not self.learn_eps:
                    pad.append(j+start_idx[i])
                padded_neighbors.append(pad)

            padded_neighbor_list.extend(padded_neighbors)
        #print(padded_neighbor_list)
        return torch.LongTensor(padded_neighbor_list)
    
    def __preprocess_neighbors_sumavepool(self, batch_graph):

        edge_mat_list = []
        start_idx = [0]

        for i, graph in enumerate(batch_graph):
            start_idx.append(start_idx[i] + len(graph.g))
            edge_mat_list.append(graph.edge_mat+start_idx[i])
        Adj_block_idx = torch.cat(edge_mat_list, 1)
        Adj_block_elem = torch.ones(Adj_block_idx.shape[1])

        #print(Adj_block_idx.shape)

        if not self.learn_eps: # add self-loop
            num_node = start_idx[-1]
            self.loop_edge = torch.LongTensor([range(num_node), range(num_node)])
            elem = torch.ones(num_node)
            Adj_block_idx = torch.cat([Adj_block_idx, self.loop_edge], 1)
            Adj_block_elem = torch.cat([Adj_block_elem, elem], 0)
        Adj_block = torch.sparse.FloatTensor(Adj_block_idx, Adj_block_elem, torch.Size([start_idx[-1], start_idx[-1]]))
        return Adj_block.to(self.device)

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
    
    def maxpool(self, h, padded_neighbor_list):

        dummy = torch.min(h, dim=0)[0]
        h_with_dummy = torch.cat([h, dummy.reshape((1, -1)).to(self.device)])
        #print(torch.max(h_with_dummy[[1, 5 , -1, -1]], dim=1))
        pooled_rep = torch.max(h_with_dummy[padded_neighbor_list], dim=1)[0]
        return pooled_rep
    
    # def next_layer_eps0(self, h, layer, padded_neighbor_list = None, Adj_block = None):

    #     if self.neighbor_pooling_type == "max":
    #         pooled = self.maxpool(h, padded_neighbor_list)
    #     else:
    #         # If sum or average pooling
    #         pooled = torch.spmm(Adj_block, h)
    #         if self.neighbor_pooling_type == "average":
    #             degree = torch.spmm(Adj_block, torch.ones((Adj_block.shape[0], 1)).to(self.device))
                
    #             pooled = pooled/degree # element-wise div
    #     pooled = pooled + (1 +self.eps0[layer])*h
       
    #     pooled_rep = self.mlps0[layer](pooled)
    #     h = self.batch_norms0[layer](pooled_rep)
    #     #print(h.device)
    #     h  = F.relu(h)
    #     return h 
    def next_layer_eps1(self, h, layer, padded_neighbor_list = None, Adj_block = None):

        if self.neighbor_pooling_type == "max":
            pooled = self.maxpool(h, padded_neighbor_list)
        else:
            # If sum or average pooling
            pooled = torch.spmm(Adj_block, h)
            #h = self.CtoD(h, layer)
            if self.neighbor_pooling_type == "average":
                degree = torch.spmm(Adj_block, torch.ones((Adj_block.shape[0], 1)).to(self.device))
                
                pooled = pooled/degree # element-wise div
        pooled = pooled + (1 +self.eps1[layer])*h
       
        pooled_rep = self.mlps2[layer](pooled)
        h = self.batch_norms2[layer](pooled_rep)
        #print(h.device)
        h  = F.relu(h)
        return h
    def next_layer_eps2(self, h, layer, padded_neighbor_list = None, Adj_block = None):

        if self.neighbor_pooling_type == "max":
            pooled = self.maxpool(h, padded_neighbor_list)
        else:
            # If sum or average pooling
            #h_ori = h
            #h = self.CtoD(h, layer)
            pooled = torch.spmm(Adj_block, h)
            if self.neighbor_pooling_type == "average":
                degree = torch.spmm(Adj_block, torch.ones((Adj_block.shape[0], 1)).to(self.device))
                
                pooled = pooled/degree # element-wise div
        pooled = pooled + (1 +self.eps2[layer])*h
       
        pooled_rep = self.mlps2[layer](pooled)
        h = self.batch_norms2[layer](pooled_rep)
        #print(h.device)
        h  = F.relu(h)
        return h
    
    # def next_layer0(self, h, layer, padded_neighbor_list =None, Adj_block=None):
    #     if self.neighbor_pooling_type == "max":
    #         pooled = self.maxpool(h, padded_neighbor_list)
    #     else:
    #         pooled = torch.spmm(Adj_block, h)
    #         if self.neighbor_pooling_type == "average":
    #             degree = torch.spmm(Adj_block, torch.ones((Adj_block.shape[0], 1)).to(self.device))
    #             pooled = pooled/degree
    #     pooled_rep = self.mlps0[layer](pooled)
    #     h = self.batch_norms0[layer](pooled_rep)
    #     h = F.relu(h)
    #     return h
    
    def next_layer1(self, h, layer, padded_neighbor_list =None, Adj_block=None):
        if self.neighbor_pooling_type == "max":
            pooled = self.maxpool(h, padded_neighbor_list)
        else:
            #h = self.CtoD(h, layer)
            pooled = torch.spmm(Adj_block, h)
            
            if self.neighbor_pooling_type == "average":
                degree = torch.spmm(Adj_block, torch.ones((Adj_block.shape[0], 1)).to(self.device))
                pooled = pooled/degree
        pooled_rep = self.mlps2[layer](pooled)
        h = self.batch_norms2[layer](pooled_rep)
        h = F.relu(h)
        return h
    
    def next_layer2(self, h, layer, padded_neighbor_list =None, Adj_block=None):
        if self.neighbor_pooling_type == "max":
            pooled = self.maxpool(h, padded_neighbor_list)
        else:
            pooled = torch.spmm(Adj_block, h)
            if self.neighbor_pooling_type == "average":
                degree = torch.spmm(Adj_block, torch.ones((Adj_block.shape[0], 1)).to(self.device))
                pooled = pooled/degree
        pooled_rep = self.mlps2[layer](pooled)
        h = self.batch_norms2[layer](pooled_rep)
        h = F.relu(h)
        return h
    



    # def CtoD(self, h, layer):

        
    #     h =F.gumbel_softmax(h, dim=1, tau=0.67)
    #     # h = torch.matmul(h, S)
    #     # if self.training:
    #     #     h =F.gumbel_softmax(h, dim=1)
    #     # else:
    #     #     h =F.gumbel_softmax(h, hard = True, dim=1)
    #         # max_value, max_idx = torch.max(h, dim=1, keepdim=True)
    #         # h = torch.zeros_like(h).scatter_(1, max_idx, max_value)
    #         #h.scatter_(1, max_idx.unsqueeze(1), 1)
    #     return h
    # def CtoD(self, alpha):
    #     """
    #     Samples from a gumbel-softmax distribution using the reparameterization
    #     trick.

    #     Parameters
    #     ----------
    #     alpha : torch.Tensor
    #         Parameters of the gumbel-softmax distribution. Shape (N, D)
    #     """
    #     EPS = 1e-12
    #     if self.training:
    #         # Sample from gumbel distribution
    #         unif = torch.rand(alpha.size())
    #         unif = unif.to(self.device)
    #         gumbel = -torch.log(-torch.log(unif + EPS) + EPS)
    #         # Reparameterize to create gumbel softmax sample
    #         log_alpha = torch.log(alpha + EPS)
    #         logit = (log_alpha + gumbel) / self.temperature

    #         return F.softmax(logit, dim=1)
    #     else:
    #         # In reconstruction mode, pick most likely sample
    #         _, max_alpha = torch.max(alpha, dim=1)
    #         one_hot_samples = torch.zeros(alpha.size())
    #         # On axis 1 of one_hot_samples, scatter the value 1 at indices
    #         # max_alpha. Note the view is because scatter_ only accepts 2D
    #         # tensors.
    #         one_hot_samples.scatter_(1, max_alpha.view(-1, 1).data.cpu(), 1)

    #         one_hot_samples = one_hot_samples.to(self.device)
    #         return one_hot_samples
            
    # def compute_d(self, hidden_h2, device):
    #     h2 = hidden_h2[0]
    #     h2_inx = hidden_h2[1]
    #     classes = h2_inx.size(1)

    #     # Prepare indices for positive and negative samples
    #     p_inx_list = [torch.nonzero(h2_inx[:, i], as_tuple=False).squeeze(1).tolist() for i in range(classes)]
    #     n_inx_list = [[j for j in range(h2.size(0)) if j not in p_inx] for p_inx in p_inx_list]

    #     Q = 10
    #     loss = 0
    #     # for i in range(len(p_inx_list)):
    #     #     print(str(i)+':'+str(len(p_inx_list[i])))

    #     for p_inx, n_inx in zip(p_inx_list, n_inx_list):
    #         if not p_inx:
    #             continue
           
    #         if len(p_inx) == 1:
    #             continue

    #         node_scores = []
    #         node_inx = p_inx[0]
    #         pos_scores = [torch.log(torch.sigmoid(F.cosine_similarity(h2[node_inx], h2[node_inx2], dim=0))) 
    #                           for node_inx2 in p_inx if node_inx2 != node_inx]

    #         n_row = random.sample(n_inx, min(Q, len(n_inx)))
    #         neg_scores = [torch.log(torch.sigmoid(-F.cosine_similarity(h2[node_inx], h2[node_inx3], dim=0))) 
    #                           for node_inx3 in n_row]
    #         if len(neg_scores) == 0:
    #             continue

    #         pos_scores = torch.mean(torch.stack(pos_scores))
    #         neg_scores = Q * torch.mean(torch.stack(neg_scores))
    #         node_scores.append(-pos_scores - neg_scores)

    #         loss += torch.mean(torch.stack(node_scores))

    #     average_loss = loss / len(p_inx_list) if p_inx_list else torch.tensor(0.0, device=device)
    #     #print(average_loss)
    #     return average_loss
        
    # def h1_trans(self, h1, layer):
    #     if layer == self.num_layers -2: 
    #         h1 = h1
    #     else:
    #         h1 = self.linear_softmax[layer](h1)
    #     return h1
    
    def pairnorm(self, X, s=0.0005):
        n, d =  X.shape
        X_mean = torch.mean(X, dim=0)
        X_c = X - X_mean
        X_norm_squared = torch.norm(X_c, p='fro')**2  # Frobenius norm squared
        X_scaled = s * torch.sqrt(torch.tensor(n)) * (X_c / torch.sqrt(X_norm_squared))
        return X_scaled
        
    def forward(self, batch_graph):
        X_concat = torch.cat([graph.node_features for graph in batch_graph], 0).to(self.device)

        #hidden_rep = [X_concat] 
        #h1 = X_concat
        
        h2 = X_concat
        # h0_1 = X_concat
        # h0_2 = X_concat
        # h_com1 = X_concat
        # h_com2 = X_concat

        #h_total = torch.cat((h1, h2), dim=1)
        #print(h_total.shape)
        hidden_rep = [X_concat]
        #print(h2.shape)
        alpha = []
      

        if self.neighbor_pooling_type == "max":
            padded_neighbor_list = self.__preprocess_neighbors_maxpool(batch_graph)
        else:
            
            Adj_block = self.__preprocess_neighbors_sumavepool(batch_graph)
            #print(Adj_block.shape)

        for layer in range(self.num_layers-1):
            
            h2_ori = h2
            
            if self.neighbor_pooling_type == "max" and self.learn_eps:
                # h0_1 = self.next_layer_eps0(h0_1, layer, padded_neighbor_list = padded_neighbor_list)
                # h0_2 = self.next_layer_eps0(h0_2, layer, padded_neighbor_list = padded_neighbor_list)
                #h1 = self.next_layer_eps1(h1, layer, padded_neighbor_list = padded_neighbor_list)
                h2 = self.next_layer_eps2(h2, layer, padded_neighbor_list = padded_neighbor_list)
            elif not self.neighbor_pooling_type == "max" and self.learn_eps:
                # h0_1 = self.next_layer_eps0(h0_1, layer, Adj_block=Adj_block)
                # h0_2 = self.next_layer_eps0(h0_2, layer, Adj_block=Adj_block)
                #h1 = self.next_layer_eps1(h1, layer, Adj_block=Adj_block)
                h2 = self.next_layer_eps2(h2, layer, Adj_block=Adj_block)
            elif self.neighbor_pooling_type == 'max' and not self.learn_eps:
                # h0_1 = self.next_layer0(h0_1, layer, padded_neighbor_list = padded_neighbor_list)
                # h0_2 = self.next_layer0(h0_2, layer, padded_neighbor_list = padded_neighbor_list)
                #h1 = self.next_layer1(h1, layer, padded_neighbor_list = padded_neighbor_list)
                h2 = self.next_layer2(h2, layer, padded_neighbor_list = padded_neighbor_list)
            elif not self.neighbor_pooling_type == "max" and not self.learn_eps:
                # h0_1 = self.next_layer0(h0_1, layer, Adj_block=Adj_block)
                # h0_2 = self.next_layer0(h0_2, layer, Adj_block=Adj_block)
                #h1 = self.next_layer1(h1, layer, Adj_block = Adj_block)
                h2 = self.next_layer2(h2, layer, Adj_block = Adj_block)
            #h_com = (h0_1 + h0_2)/2
            # print(h1.shape)
            # print(h2.shape)
            #h_t = torch.stack([h1, h2], dim=1)
            #print(h_t.shape)
            #h_tmp, att = self.attention[layer](h_t)
            #h_tmp = torch.cat((h1, h2), dim=1)

            # for common loss
            # h_com1 = torch.cat((h_com1, h0_1), dim=1)
            # h_com2 = torch.cat((h_com2, h0_2), dim=1)
            #print(h_tmp.shape)
            #h_total = torch.cat((h_total, h_tmp), dim=1) #multi-scale
            #print(h_total.shape)
            #h_total, attn = self.attention(h_total)
            #print(attn.shape)
            #print(h_total.shape)
            
            hidden_rep.append(h2)
            h2 = self.pairnorm(h2)
            #alpha.append(h2)
            # h2 = self.CtoD(h2, layer)
            
            #h1 = self.h1_trans(h1, layer)

            
            
            #h0_2 = self.CtoD(h0_2, layer)
        #print(hidden_rep1)
        #h_total = torch.stack([hidden_rep1, hidden_rep2], dim=1)
      
        
        #print(h_total.shape)
        #print(attn)
        #h_emb = [X_concat, h_total]


        score_over_layer = 0
        graph_pool = self.__preprocess_graphpool(batch_graph)

        for layer, h in enumerate(hidden_rep):
            #print(h.shape)
            pooled_h = torch.spmm(graph_pool, h) # len(batch_graph)* dim of feature
            score_over_layer +=F.dropout(self.linears_prediction[layer](pooled_h), self.final_dropout, training=self.training)
            #print(score_over_layer)
        
        return score_over_layer
    
        
