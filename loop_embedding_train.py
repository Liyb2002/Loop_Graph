import Preprocessing.dataloader_loop_embedding
import Preprocessing.gnn_graph
import Encoders.gnn.gnn
import Encoders.helper

from torch.utils.data import DataLoader, random_split, Subset
from tqdm import tqdm
from Preprocessing.config import device
import torch
import torch.nn as nn
import torch.optim as optim
import os
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import Models.loop_embeddings


loop_embed_model = Models.loop_embeddings.LoopEmbeddingNetwork()
loop_decoder_model = Models.loop_embeddings.LoopConnectivityDecoder()

loop_embed_model.to(device)
loop_decoder_model.to(device)

criterion = nn.BCELoss()
optimizer = optim.Adam(list(loop_embed_model.parameters()) + list(loop_decoder_model.parameters()), lr=0.0004)

# ------------------------------------------------------------------------------# 

current_dir = os.getcwd()
save_dir = os.path.join(current_dir, 'checkpoints', 'loop_embedding_model')
os.makedirs(save_dir, exist_ok=True)

def load_models():
    loop_embed_model.load_state_dict(torch.load(os.path.join(save_dir, 'loop_embed_model.pth')))
    loop_decoder_model.load_state_dict(torch.load(os.path.join(save_dir, 'loop_decoder_model.pth')))


def save_models():
    torch.save(loop_embed_model.state_dict(), os.path.join(save_dir, 'loop_embed_model.pth'))
    torch.save(loop_decoder_model.state_dict(), os.path.join(save_dir, 'loop_decoder_model.pth'))


# ------------------------------------------------------------------------------# 



def train():
    # Load the dataset
    dataset = Preprocessing.dataloader_loop_embedding.Program_Graph_Dataset('dataset/test')
    print(f"Total number of shape data: {len(dataset)}")

    # Split the dataset into training and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create DataLoaders for training and validation
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=Preprocessing.dataloader_loop_embedding.custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=True, collate_fn=Preprocessing.dataloader_loop_embedding.custom_collate_fn)
    
    best_val_loss = float('inf')
    epochs = 10

    for epoch in range(epochs):
        loop_embed_model.train()
        loop_decoder_model.train()
        epoch_loss = 0

        # Training Loop
        for loop_features, loop_neighboring_combined, mask_loop_features in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Training"):

            # Zero the gradients
            optimizer.zero_grad()

            # Compute Loop embeddings
            stroke_loop_embeddings = loop_embed_model(loop_features, mask_loop_features)

            # Decode to predict connectivity matrix
            connectivity_matrix = loop_decoder_model(stroke_loop_embeddings, mask_loop_features)

            # Compute the loss
            loss = criterion(connectivity_matrix, loop_neighboring_combined)

            # Backpropagation and optimization
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_epoch_loss = epoch_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}], Training Loss: {avg_epoch_loss:.4f}")
        
        if avg_epoch_loss < best_val_loss:
            best_val_loss = avg_epoch_loss
            save_models()


#---------------------------------- Public Functions ----------------------------------#

train()