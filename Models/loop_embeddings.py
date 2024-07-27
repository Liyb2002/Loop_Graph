
import torch
import torch.nn as nn

from torch.nn import TransformerEncoder, TransformerEncoderLayer


class StrokeEmbeddingNetwork(nn.Module):
    def __init__(self, input_dim=6, embedding_dim=16):
        super(StrokeEmbeddingNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.fc2 = nn.Linear(32, embedding_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class LoopEmbeddingNetwork(nn.Module):
    def __init__(self, stroke_embedding_dim=16, hidden_dim=32, output_dim=32):
        super(LoopEmbeddingNetwork, self).__init__()
        self.stroke_embedding = StrokeEmbeddingNetwork(input_dim=6, embedding_dim=stroke_embedding_dim)
        self.self_attention = nn.MultiheadAttention(embed_dim=stroke_embedding_dim, num_heads=2, batch_first=True)
        self.fc = nn.Linear(stroke_embedding_dim, hidden_dim)
        self.fc_output = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, node_features, stroke_to_loop):
        face_embeddings = []
        
        for indices in stroke_to_loop:
            indices = [index.item() for index in indices]

            strokes = node_features[indices]  # shape: (num_indices, 6)
            
            # Embed the strokes
            embedded_strokes = self.stroke_embedding(strokes)  # shape: (num_indices, 16)
            
            # Add batch dimension for self-attention
            embedded_strokes = embedded_strokes.unsqueeze(0)  # shape: (1, num_indices, 16)
            
            # Apply self-attention
            attention_output, _ = self.self_attention(embedded_strokes, embedded_strokes, embedded_strokes)  # shape: (1, num_indices, 16)
            
            # Pass through fully connected layers and activation
            x = self.relu(self.fc(attention_output))  # shape: (1, num_indices, 32)
            
            # Mean pooling over strokes
            x = x.mean(dim=1)  # shape: (1, 32)
            
            # Final output layer
            face_embedding = self.fc_output(x)  # shape: (1, 32)
            face_embeddings.append(face_embedding)

        # Stack the embeddings for all faces to form the output tensor
        face_embeddings = torch.cat(face_embeddings, dim=0)  # shape: (len(stroke_to_loop), 32)

        print("face_embeddings", face_embeddings.shape)
        return face_embeddings

