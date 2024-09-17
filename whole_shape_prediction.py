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
    dataset = Preprocessing.dataloader.Program_Graph_Dataset('dataset/simple')
    print(f"Total number of shape data: {len(dataset)}")

    train_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    best_val_loss = float('inf')
    epochs = 1

    graphs = []
    loop_selection_masks = []

    # Preprocess and build the graphs
    for data in dataset:
        # Extract the necessary elements from the dataset
        stroke_cloud_loops, stroke_node_features, loop_neighboring_vertical, loop_neighboring_horizontal, stroke_to_brep, stroke_operations_order_matrix, final_brep_edges = data

        second_last_column = stroke_operations_order_matrix[:, -2].reshape(-1, 1)
        chosen_strokes = (second_last_column == 1).nonzero(as_tuple=True)[0]  # Indices of chosen strokes
        loop_chosen_mask = []
        for loop in stroke_cloud_loops:
            if all(stroke in chosen_strokes for stroke in loop):
                loop_chosen_mask.append(1)  # Loop is chosen
            else:
                loop_chosen_mask.append(0)  # Loop is not chosen
        
        loop_chosen_mask_tensor = torch.tensor(loop_chosen_mask).reshape(-1, 1)
        if not (loop_chosen_mask_tensor == 1).any():
            continue

        # Build the graph
        gnn_graph = Preprocessing.gnn_graph.SketchLoopGraph(
            stroke_cloud_loops, 
            stroke_node_features, 
            loop_neighboring_vertical, 
            loop_neighboring_horizontal, 
            stroke_to_brep
        )

        # Prepare the pair
        graphs.append(gnn_graph)
        loop_selection_masks.append(loop_chosen_mask_tensor)

    print(f"Total number of preprocessed graphs: {len(graphs)}")



    # Training loop
    for epoch in range(epochs):
        total_loss = 0.0
        for gnn_graph, loop_selection_mask in zip(graphs, loop_selection_masks):

            Encoders.helper.vis_whole_graph(gnn_graph, loop_selection_mask)




#---------------------------------- Public Functions ----------------------------------#

train()