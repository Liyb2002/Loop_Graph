import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GCNConv, EdgeConv
from torch_geometric.data import HeteroData

import Encoders.gnn.basic

class SemanticModule(nn.Module):
    def __init__(self, in_channels=11):
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


class Fillet_Decoder(nn.Module):
    def __init__(self, hidden_channels=256):
        super(Fillet_Decoder, self).__init__()

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


class Chamfer_Decoder(nn.Module):
    def __init__(self, hidden_channels=256):
        super(Chamfer_Decoder, self).__init__()

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
    def __init__(self, embed_dim=128, num_heads=8, ff_dim=256, num_classes=10, dropout=0.1):
        super(Program_Decoder, self).__init__()
        # Cross-attention layer
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)

        # Feed-forward layers and normalization
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim * 2, num_classes)  # Adjusting for concatenated output
        
        # Program encoder
        self.program_encoder = ProgramEncoder()

    def forward(self, x_dict, program_tokens):
        # Encode the program tokens to get their embeddings
        program_embedding = self.program_encoder(program_tokens)  # (batch_size, seq_len, embed_dim)

        # Process stroke node embeddings
        num_strokes = x_dict['stroke'].shape[0]
        batch_size_stroke = max(1, num_strokes // 400)  # Ensure batch_size is at least 1
        node_features_stroke = x_dict['stroke'].view(batch_size_stroke, min(400, num_strokes), 128)
        node_features_stroke = node_features_stroke.transpose(0, 1)  # Transpose for MultiheadAttention

        # Process loop node embeddings
        num_loops = x_dict['loop'].shape[0]
        batch_size_loop = max(1, num_loops // 400)  # Ensure batch_size is at least 1
        node_features_loop = x_dict['loop'].view(batch_size_loop, min(400, num_loops), 128)
        node_features_loop = node_features_loop.transpose(0, 1)  # Transpose for MultiheadAttention

        # Transpose program embeddings
        program_embedding = program_embedding.transpose(0, 1)  # (seq_len, batch_size, embed_dim)

        # Cross-attention for stroke nodes
        attn_output_stroke, _ = self.cross_attn(program_embedding, node_features_stroke, node_features_stroke)
        out_stroke = self.norm1(program_embedding + attn_output_stroke)
        ff_output_stroke = self.ff(out_stroke)
        out_stroke = self.norm2(out_stroke + ff_output_stroke)
        out_stroke = self.dropout(out_stroke)
        cls_attn_stroke, _ = self.self_attn(out_stroke, out_stroke, out_stroke)
        cls_stroke = out_stroke[0]  # Take the CLS token representation

        # Cross-attention for loop nodes
        attn_output_loop, _ = self.cross_attn(program_embedding, node_features_loop, node_features_loop)
        out_loop = self.norm1(program_embedding + attn_output_loop)
        ff_output_loop = self.ff(out_loop)
        out_loop = self.norm2(out_loop + ff_output_loop)
        out_loop = self.dropout(out_loop)
        cls_attn_loop, _ = self.self_attn(out_loop, out_loop, out_loop)

        cls_loop = out_loop[0]  # Take the CLS token representation

        # Concatenate outputs from stroke and loop
        combined_output = torch.cat([cls_stroke, cls_loop], dim=-1)

        # Classification
        logits = self.classifier(combined_output)

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
    def __init__(self, vocab_size=20, embedding_dim=128):
        super(ProgramEncoder, self).__init__()
        # Set padding_idx to -1 to ignore -1 in embeddings
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=-1)

    def forward(self, x):
        return self.embedding(x)
