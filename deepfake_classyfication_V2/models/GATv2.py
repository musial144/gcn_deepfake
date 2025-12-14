import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool
from torch_geometric.data import Data


""" Definiujemy model GATv2, który z cech wyodrębnionych analizuje wskazane cechy wyesktrachowane z listy patchy obrazu
"""
class GATv2(nn.Module):
    def __init__(self, in_dim, hid=256, out_dim=2, heads1=4, heads2=4, dropout=0.5):
        super().__init__()
        self.out_dim = out_dim
        # self.conv1 = GATv2Conv(in_dim, hid, heads=heads1, dropout=dropout, add_self_loops=False)
        self.conv1 = GATv2Conv(in_dim, hid, heads=heads1, add_self_loops=False)
        # self.conv2 = GATv2Conv(hid*heads1, hid, heads=heads2, dropout=dropout, add_self_loops=False)
        self.conv2 = GATv2Conv(hid*heads1, hid, heads=heads2, add_self_loops=False)
        self.lin = nn.Sequential(
            # nn.ELU(),
            nn.SiLU(),
            # nn.Dropout(dropout),
            nn.Linear(hid*heads2, hid),
            # nn.ELU(),
            nn.SiLU(),
            # nn.Dropout(dropout),
            # nn.Linear(hid*heads2, out_dim),
            nn.Linear(hid, out_dim)
        )

    def forward(self, data: Data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        h = self.conv1(x, edge_index)   
        # h = F.elu(h)
        h = F.silu(h)
        # h = F.dropout(h, p=self.conv1.dropout, training=self.training)
        h = self.conv2(h, edge_index)   
        # h = F.elu(h)
        # h = F.silu(h)
        # h = F.dropout(h, p=self.conv2.dropout, training=self.training)
        g = global_mean_pool(h, batch)  
        return g #self.lin(g) 