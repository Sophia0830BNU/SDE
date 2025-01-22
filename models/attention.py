import torch
import torch.nn as nn
class Attention(nn.Module):
    def __init__(self, in_size, hidden_size=16):
        super(Attention, self).__init__()

        self.attention_list = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )
    def forward(self, x):
        #print(x.shape)
        w = self.attention_list(x)
        #print(w.shape)
        att = torch.softmax(w, dim=1) # shape: [batch_size, number of feature kinds, 1]
        # out = torch.reshape(torch.flatten(att*x, start_dim=1), (x.shape[0], x.shape[-1]*x.shape[1]))  # multi-scale
        out = (att * x).sum(1)
        return out,att