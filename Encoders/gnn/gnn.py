import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GCNConv, EdgeConv
from torch_geometric.data import HeteroData

import Encoders.gnn.basic

class SemanticModule(nn.Module):
    def __init__(self, in_channels=9):
        super(SemanticModule, self).__init__()
        self.local_head = Encoders.gnn.basic.GeneralHeteroConv(['represents_sum', 'represented_by_sum', 'neighboring_vertical_mean', 'neighboring_horizontal_mean', 'contains_sum', 'order_add', 'perpendicular_mean'], in_channels, 16)

        self.layers = nn.ModuleList([
            Encoders.gnn.basic.ResidualGeneralHeteroConvBlock(['represents_sum', 'represented_by_sum', 'neighboring_vertical_mean', 'neighboring_horizontal_mean', 'contains_sum', 'order_add', 'perpendicular_mean'], 16, 32),
            Encoders.gnn.basic.ResidualGeneralHeteroConvBlock(['represents_sum', 'represented_by_sum', 'neighboring_vertical_mean', 'neighboring_horizontal_mean', 'contains_sum', 'order_add', 'perpendicular_mean'], 32, 64),
            Encoders.gnn.basic.ResidualGeneralHeteroConvBlock(['represents_sum', 'represented_by_sum', 'neighboring_vertical_mean', 'neighboring_horizontal_mean', 'contains_sum', 'order_add', 'perpendicular_mean'], 64, 128),
            Encoders.gnn.basic.ResidualGeneralHeteroConvBlock(['represents_sum', 'represented_by_sum', 'neighboring_vertical_mean', 'neighboring_horizontal_mean', 'contains_sum', 'order_add', 'perpendicular_mean'], 128, 128),
            Encoders.gnn.basic.ResidualGeneralHeteroConvBlock(['represents_sum', 'represented_by_sum', 'neighboring_vertical_mean', 'neighboring_horizontal_mean', 'contains_sum', 'order_add', 'perpendicular_mean'], 128, 128),
            Encoders.gnn.basic.ResidualGeneralHeteroConvBlock(['represents_sum', 'represented_by_sum', 'neighboring_vertical_mean', 'neighboring_horizontal_mean', 'contains_sum', 'order_add', 'perpendicular_mean'], 128, 128),

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


class Program_Decoder(nn.Module):
    def __init__(self, method_type, embed_dim=128, num_heads=8, ff_dim=256, num_classes=10, dropout=0.1):
        super(Program_Decoder, self).__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes)

        self.program_encoder = ProgramEncoder()
        self.method_type = method_type

    def forward(self, x_dict, program_tokens):
        # Encode the program tokens to get their embeddings
        program_embedding = self.program_encoder(program_tokens)  # (batch_size, seq_len, embed_dim)

        # Reshape x_dict['stroke'] to (batch_size, num_strokes, embed_dim) -> (batch_size, 200, 128)

        if self.method_type == 'stroke':
            node_features = x_dict['stroke']
        
        elif self.method_type == 'loop':
            node_features = x_dict['loop']

        # Perform cross-attention between program_embedding (query) and stroke_features (key, value)
        attn_output, _ = self.cross_attn(query=program_embedding, key=node_features, value=node_features)

        # Add residual connection and normalize
        out = self.norm1(program_embedding + attn_output)

        # Apply feed-forward layer with residual connection and normalization
        ff_output = self.ff(out)
        out = self.norm2(out + ff_output)

        # Apply dropout for regularization
        out = self.dropout(out)

        # Pool the output (e.g., mean over the sequence) before classification
        out_mean = out.mean(dim=0)  # Mean over the sequence dimension (seq_len)

        # Final classification logits
        logits = self.classifier(out_mean)

        return logits


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


class ProgramEncoder(nn.Module):
    def __init__(self, vocab_size=20, embedding_dim=16, hidden_dim=128):
        super(ProgramEncoder, self).__init__()
        # Set padding_idx to -1 to ignore -1 in embeddings
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=-1)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        return lstm_out
