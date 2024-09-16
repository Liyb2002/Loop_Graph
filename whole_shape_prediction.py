import Preprocessing.dataloader
import Preprocessing.gnn_graph

import Encoders.gnn.gnn
import Encoders.helper

from torch_geometric.loader import DataLoader

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


graph_encoder = Encoders.gnn.gnn.SemanticModule()
graph_encoder.to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(list(graph_encoder.parameters()), lr=0.0004)

# ------------------------------------------------------------------------------# 

current_dir = os.getcwd()
save_dir = os.path.join(current_dir, 'checkpoints', 'sketch_prediction')
os.makedirs(save_dir, exist_ok=True)

def load_models():
    graph_encoder.load_state_dict(torch.load(os.path.join(save_dir, 'graph_encoder.pth')))


def save_models():
    torch.save(graph_encoder.state_dict(), os.path.join(save_dir, 'graph_encoder.pth'))


# ------------------------------------------------------------------------------# 



def train():
    # Load the dataset
    dataset = Preprocessing.dataloader.Program_Graph_Dataset('dataset/test')
    print(f"Total number of shape data: {len(dataset)}")

    train_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    best_val_loss = float('inf')
    epochs = 1


    graphs = []
    for data in dataset:
        # Extract the necessary elements from the dataset
        stroke_cloud_loops, stroke_node_features, loop_embeddings, loop_neighboring_vertical, loop_neighboring_horizontal, stroke_to_brep, stroke_operations_order_matrix, final_brep_edges= data

        # Squeeze the tensors if needed
        loop_embeddings = loop_embeddings.squeeze(0)
        loop_neighboring_vertical = loop_neighboring_vertical.squeeze(0)
        loop_neighboring_horizontal = loop_neighboring_horizontal.squeeze(0)
        stroke_to_brep = stroke_to_brep.squeeze(0)

        # Build the graph
        gnn_graph = Preprocessing.gnn_graph.SketchHeteroData(loop_embeddings, loop_neighboring_vertical, loop_neighboring_horizontal, stroke_to_brep)

        # Move graph to GPU
        gnn_graph = gnn_graph.to(device)
        graphs.append(gnn_graph)

        Encoders.helper.vis_partial_graph(stroke_cloud_loops, stroke_node_features, stroke_to_brep)
        Encoders.helper.vis_whole_graph(stroke_cloud_loops, stroke_node_features)
        Encoders.helper.vis_brep(final_brep_edges)


    print(f"Total number of preprocessed graphs: {len(graphs)}")


    best_val_loss = float('inf')
    epochs = 1

    for epoch in range(epochs):
        for gnn_graph in tqdm(graphs, desc=f"Epoch {epoch+1}/{epochs} - Training"):

            print("gnn_graph", gnn_graph['stroke'].x.shape)

#---------------------------------- Public Functions ----------------------------------#

train()