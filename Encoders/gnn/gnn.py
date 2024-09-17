import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GCNConv, EdgeConv
from torch_geometric.data import HeteroData

import Encoders.gnn.basic

class SemanticModule(nn.Module):
    def __init__(self, in_channels=6):
        super(SemanticModule, self).__init__()
        self.local_head = Encoders.gnn.basic.GeneralHeteroConv(['representedBy_sum', 'neighboring_mean', 'order_add'], in_channels, 32)

        self.layers = nn.ModuleList([
            Encoders.gnn.basic.ResidualGeneralHeteroConvBlock(['representedBy_sum', 'neighboring_mean', 'order_add'], 32, 32),
            Encoders.gnn.basic.ResidualGeneralHeteroConvBlock(['representedBy_sum', 'neighboring_mean', 'order_add'], 32, 32),
            Encoders.gnn.basic.ResidualGeneralHeteroConvBlock(['representedBy_sum', 'neighboring_mean', 'order_add'], 32, 32),
            Encoders.gnn.basic.ResidualGeneralHeteroConvBlock(['representedBy_sum', 'neighboring_mean', 'order_add'], 32, 32)
        ])


    def forward(self, x_dict, edge_index_dict):

        x_dict = self.local_head(x_dict, edge_index_dict)

        for layer in self.layers:
            x_dict = layer(x_dict, edge_index_dict)
        
        x_dict = {key: x.relu() for key, x in x_dict.items()}

        return x_dict




class Sketch_prediction(nn.Module):
    def __init__(self, hidden_channels=64):
        super(Sketch_prediction, self).__init__()

        self.decoder = nn.Sequential(
            nn.Linear(32, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, 1),
        )

    def forward(self, x_dict):
        return torch.sigmoid(self.decoder(x_dict['loop']))
