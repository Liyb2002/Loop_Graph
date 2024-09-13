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

loop_embed_model.to(device)

criterion = nn.BCELoss()
optimizer = optim.Adam(list(loop_embed_model.parameters()), lr=0.0004)

# ------------------------------------------------------------------------------# 

current_dir = os.getcwd()
save_dir = os.path.join(current_dir, 'checkpoints', 'loop_embedding_model')
os.makedirs(save_dir, exist_ok=True)

def load_models():
    loop_embed_model.load_state_dict(torch.load(os.path.join(save_dir, 'loop_embed_model.pth')))


def save_models():
    torch.save(loop_embed_model.state_dict(), os.path.join(save_dir, 'loop_embed_model.pth'))


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
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)
    
    best_val_loss = float('inf')
    epochs = 1


    for epoch in range(epochs):

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Training"):
            stroke_cloud_loops, stroke_node_features = batch

            stroke_node_features = stroke_node_features.to(torch.float32).squeeze(0)

            # Loop embeddings
            stroke_loop_embeddings = loop_embed_model(stroke_node_features, stroke_cloud_loops)

            print("stroke_loop_embeddings", stroke_loop_embeddings.shape)



#---------------------------------- Public Functions ----------------------------------#

train()