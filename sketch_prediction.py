import Preprocessing.dataloader
import Preprocessing.gnn_graph
import Encoders.gnn.gnn
import Encoders.helper

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

import Models.loop_embeddings

# Load the dataset
dataset = Preprocessing.dataloader.Program_Graph_Dataset('dataset/test')
good_data_indices = [i for i, data in enumerate(dataset) if data[0][-1] == 1]
filtered_dataset = Subset(dataset, good_data_indices)
print(f"Total number of sketch data: {len(filtered_dataset)}")

# Split the dataset into training and validation sets
train_size = int(0.8 * len(filtered_dataset))
val_size = len(filtered_dataset) - train_size
train_dataset, val_dataset = random_split(filtered_dataset, [train_size, val_size])

# Create DataLoaders for training and validation
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)


loop_embed_model = Models.loop_embeddings.LoopEmbeddingNetwork()
graph_encoder = Encoders.gnn.gnn.SemanticModule()
graph_decoder = Encoders.gnn.gnn.Sketch_brep_prediction()


def train():
    best_val_loss = float('inf')
    epochs = 30

    for epoch in range(epochs):
        total_train_loss = 0.0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Training"):
            program, node_features, operations_order_matrix, gnn_strokeCloud_edges, face_to_stroke, brep_to_stroke, edge_features, gnn_brep_edges, brep_stroke_connection = batch            
        
            if len(edge_features) == 0:
                continue

            node_features = node_features.to(torch.float32).to(device).squeeze(0)
            edge_features = torch.tensor(edge_features)
            edge_features = edge_features.to(torch.float32).to(device)
            gnn_strokeCloud_edges = gnn_strokeCloud_edges.to(torch.float32).to(device).squeeze(0)
            gnn_brep_edges = gnn_brep_edges.to(torch.float32).to(device).squeeze(0)
            brep_stroke_connection = brep_stroke_connection.to(torch.float32).to(device).squeeze(0)


            sketch_loop_embeddings = loop_embed_model(node_features, face_to_stroke)
            brep_loop_embeddings = loop_embed_model(edge_features, brep_to_stroke)

            # Build graph
            gnn_graph = Preprocessing.gnn_graph.SketchHeteroData(sketch_loop_embeddings, brep_loop_embeddings, gnn_strokeCloud_edges, gnn_brep_edges, brep_stroke_connection)
            x_dict = graph_encoder(gnn_graph.x_dict, gnn_graph.edge_index_dict)

            # Predict
            prediction = graph_decoder(x_dict)

            # Prepare Ground_truth
            target_op_index = len(program[0]) - 1
            op_to_index_matrix = operations_order_matrix
            kth_operation = Encoders.helper.get_kth_operation(op_to_index_matrix, target_op_index).to(device)
            face_choice = Encoders.helper.stroke_to_face(kth_operation, face_to_stroke)

            # Vis
            # Encoders.helper.vis_stroke_cloud(edge_features)
            # Encoders.helper.vis_stroke_cloud(node_features)
            # Encoders.helper.vis_gt(face_choice, face_to_stroke, node_features)

            print("-----")




#---------------------------------- Public Functions ----------------------------------#

train()