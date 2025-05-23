
import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        '''
            num_layers: number of layers in mlp
            input_dim: dimensionality of input features
            hidden_dim: dimensionality of hidden units at ALL layers
            output_dim: dimensionality of output dim in mlp
        '''
        super(MLP, self).__init__()
        self.linear_or_not = True
        self.num_layers = num_layers
        
        if num_layers < 1:
            raise ValueError("number of layers should be positive")
        elif num_layers ==1:
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            
            self.batch_norms = torch.nn.ModuleList()
            
            for layer in range(num_layers-1):
                if layer == 0:
                    self.linears.append(nn.Linear(input_dim, hidden_dim))
                else:
                    self.linears.append(nn.Linear(hidden_dim, hidden_dim))
                self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

            self.linears.append(nn.Linear(hidden_dim, output_dim))
        
    def forward(self, x):
        if self.linear_or_not:
            return self.linear(x)
        else:
            h = x
            for layer in range(self.num_layers-1):
                h = F.relu(self.batch_norms[layer](self.linears[layer](h)))
            return self.linears[self.num_layers-1](h)