import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GCNConv, EdgeConv
from torch_geometric.data import HeteroData

import Encoders.gnn.basic

class SemanticModule(nn.Module):
    def __init__(self, in_channels=6):
        super(SemanticModule, self).__init__()
        self.local_head = Encoders.gnn.basic.GeneralHeteroConv(['representedBy_sum', 'neighboring_vertical_mean', 'neighboring_horizontal_mean', 'contains_mean', 'order_add'], in_channels, 16)

        self.layers = nn.ModuleList([
            Encoders.gnn.basic.ResidualGeneralHeteroConvBlock(['representedBy_sum', 'neighboring_vertical_mean', 'neighboring_horizontal_mean', 'contains_mean', 'order_add'], 16, 32),
            Encoders.gnn.basic.ResidualGeneralHeteroConvBlock(['representedBy_sum', 'neighboring_vertical_mean', 'neighboring_horizontal_mean', 'contains_mean', 'order_add'], 32, 64),
            Encoders.gnn.basic.ResidualGeneralHeteroConvBlock(['representedBy_sum', 'neighboring_vertical_mean', 'neighboring_horizontal_mean', 'contains_mean', 'order_add'], 64, 64),
            Encoders.gnn.basic.ResidualGeneralHeteroConvBlock(['representedBy_sum', 'neighboring_vertical_mean', 'neighboring_horizontal_mean', 'contains_mean', 'order_add'], 64, 64),
            Encoders.gnn.basic.ResidualGeneralHeteroConvBlock(['representedBy_sum', 'neighboring_vertical_mean', 'neighboring_horizontal_mean', 'contains_mean', 'order_add'], 64, 64),
            Encoders.gnn.basic.ResidualGeneralHeteroConvBlock(['representedBy_sum', 'neighboring_vertical_mean', 'neighboring_horizontal_mean', 'contains_mean', 'order_add'], 64, 64)

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
            nn.Linear(64, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(hidden_channels, 1),
        )

    def forward(self, x_dict):
        return torch.sigmoid(self.decoder(x_dict['loop']))



#---------------------------------- Loss Function ----------------------------------#

class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # Balances the importance of positive and negative examples
        self.gamma = gamma  # Focuses on hard examples

    def forward(self, probs, targets):        
        # Compute binary cross-entropy loss but do not reduce it
        BCE_loss = F.binary_cross_entropy(probs, targets, reduction='none')

        # Apply the focal loss scaling factor (1 - pt)^gamma
        pt = torch.exp(-BCE_loss)  # Probability of the true class
        focal_loss = self.alpha * ((1 - pt) ** self.gamma) * BCE_loss

        return focal_loss.mean()
