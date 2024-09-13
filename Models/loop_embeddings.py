
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
        if node_features.shape[1] ==0:
            return torch.zeros((1, 32), dtype=torch.float32)

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

        return face_embeddings




class LoopConnectivityDecoder(nn.Module):
    def __init__(self, embedding_dim=32, hidden_dim=64):
        super(LoopConnectivityDecoder, self).__init__()
        # A feed-forward network to predict connectivity
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, loop_embeddings):
        """
        Given loop embeddings, produce a connectivity matrix.
        Args:
            loop_embeddings (torch.Tensor): Tensor of shape (num_loops, embedding_dim)
        Returns:
            connectivity_matrix (torch.Tensor): Tensor of shape (num_loops, num_loops) with values in {0, 1}
        """
        num_loops = loop_embeddings.size(0)
        
        # Create empty matrix for storing connectivity predictions
        connectivity_matrix = torch.zeros(num_loops, num_loops, device=loop_embeddings.device)
        
        # Compute pairwise connectivity
        for i in range(num_loops):
            for j in range(i + 1, num_loops):
                # Concatenate embeddings for the pair (i, j)
                pair_embedding = torch.cat([loop_embeddings[i], loop_embeddings[j]], dim=0)  # Shape: (embedding_dim * 2)
                
                # Predict if they share an edge
                connectivity_score = self.fc(pair_embedding)  # Shape: (1,)
                
                # Store the result symmetrically
                connectivity_matrix[i, j] = connectivity_score
                connectivity_matrix[j, i] = connectivity_score
        
        return connectivity_matrix
