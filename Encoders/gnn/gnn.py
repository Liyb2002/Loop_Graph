import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GCNConv, EdgeConv
from torch_geometric.data import HeteroData

import Encoders.gnn.basic

class SemanticModule(nn.Module):
    def __init__(self, in_channels=7):
        super(SemanticModule, self).__init__()
        self.local_head = Encoders.gnn.basic.GeneralHeteroConv(['representedBy_sum', 'neighboring_vertical_mean', 'neighboring_horizontal_mean', 'contains_mean', 'coplanar_mean', 'order_add', 'connect_mean'], in_channels, 16)

        self.layers = nn.ModuleList([
            Encoders.gnn.basic.ResidualGeneralHeteroConvBlock(['representedBy_sum', 'neighboring_vertical_mean', 'neighboring_horizontal_mean', 'contains_mean', 'coplanar_mean', 'order_add', 'connect_mean'], 16, 32),
            Encoders.gnn.basic.ResidualGeneralHeteroConvBlock(['representedBy_sum', 'neighboring_vertical_mean', 'neighboring_horizontal_mean', 'contains_mean', 'coplanar_mean', 'order_add', 'connect_mean'], 32, 64),
            Encoders.gnn.basic.ResidualGeneralHeteroConvBlock(['representedBy_sum', 'neighboring_vertical_mean', 'neighboring_horizontal_mean', 'contains_mean', 'coplanar_mean', 'order_add', 'connect_mean'], 64, 128),
            Encoders.gnn.basic.ResidualGeneralHeteroConvBlock(['representedBy_sum', 'neighboring_vertical_mean', 'neighboring_horizontal_mean', 'contains_mean', 'coplanar_mean', 'order_add', 'connect_mean'], 128, 128),
            Encoders.gnn.basic.ResidualGeneralHeteroConvBlock(['representedBy_sum', 'neighboring_vertical_mean', 'neighboring_horizontal_mean', 'contains_mean', 'coplanar_mean', 'order_add', 'connect_mean'], 128, 128),
            Encoders.gnn.basic.ResidualGeneralHeteroConvBlock(['representedBy_sum', 'neighboring_vertical_mean', 'neighboring_horizontal_mean', 'contains_mean', 'coplanar_mean', 'order_add', 'connect_mean'], 128, 128)

        ])


    def forward(self, x_dict, edge_index_dict):

        x_dict = self.local_head(x_dict, edge_index_dict)

        for layer in self.layers:
            x_dict = layer(x_dict, edge_index_dict)
        
        x_dict = {key: x.relu() for key, x in x_dict.items()}

        return x_dict



class Sketch_Decoder(nn.Module):
    def __init__(self, hidden_channels=256):
        super(Sketch_Decoder, self).__init__()

        self.decoder = nn.Sequential(
            nn.Linear(128, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(hidden_channels, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(64, 1),

        )

    def forward(self, x_dict):
        return torch.sigmoid(self.decoder(x_dict['loop']))


class Extrude_Decoder(nn.Module):
    def __init__(self, hidden_channels=256):
        super(Extrude_Decoder, self).__init__()

        self.decoder = nn.Sequential(
            nn.Linear(128, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(hidden_channels, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(64, 1),

        )

    def forward(self, x_dict):
        return torch.sigmoid(self.decoder(x_dict['stroke']))


#---------------------------------- Loss Function ----------------------------------#

class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma 

    def forward(self, probs, targets):        
        # Compute binary cross-entropy loss but do not reduce it
        BCE_loss = F.binary_cross_entropy(probs, targets, reduction='none')

        pt = torch.exp(-BCE_loss)  # Probability of the true class
        focal_loss = self.alpha * ((1 - pt) ** self.gamma) * BCE_loss

        return focal_loss.mean()
