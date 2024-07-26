import Preprocessing.dataloader
import Preprocessing.gnn_graph_full


from torch.utils.data import DataLoader, random_split, Subset
from tqdm import tqdm
from config import device
import torch
import torch.nn as nn
import torch.optim as optim
import os
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# Load the dataset
dataset = Preprocessing.dataloader.Program_Graph_Dataset('dataset/test')
good_data_indices = [i for i, data in enumerate(dataset) if data[5][-1] == 1]
filtered_dataset = Subset(dataset, good_data_indices)
print(f"Total number of sketch data: {len(filtered_dataset)}")

# Split the dataset into training and validation sets
train_size = int(0.8 * len(filtered_dataset))
val_size = len(filtered_dataset) - train_size
train_dataset, val_dataset = random_split(filtered_dataset, [train_size, val_size])

# Create DataLoaders for training and validation
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)




def train():
    best_val_loss = float('inf')
    epochs = 30

    for epoch in range(epochs):
        total_train_loss = 0.0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Training"):
            node_features, operations_order_matrix, loop_features, loop_edges, face_to_stroke, program, face_boundary_points, face_feature_gnn_list, face_features, edge_features, vertex_features, edge_index_face_edge_list, edge_index_edge_vertex_list, edge_index_face_face_list, index_id = batch
            
            
            node_features = node_features.to(torch.float32).to(device).squeeze(0)
            operations_order_matrix = operations_order_matrix.to(torch.float32).to(device)
            loop_features = loop_features.to(torch.float32).to(device).squeeze(0)
            loop_edges = loop_edges.to(torch.float32).to(device).squeeze(0)

            print("loop_features", loop_features.shape) 
            print("loop_features", loop_edges.shape)




#---------------------------------- Public Functions ----------------------------------#

train()