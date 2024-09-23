import Preprocessing.dataloader
import Preprocessing.gnn_graph
import Preprocessing.gnn_graph_stroke

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

graph_encoder = Encoders.gnn.gnn.SemanticModule()
graph_decoder = Encoders.gnn.gnn.Sketch_prediction()

graph_encoder.to(device)
graph_decoder.to(device)

criterion = Encoders.gnn.gnn.FocalLoss(alpha=0.9, gamma=3.0)
optimizer = optim.Adam(list(graph_encoder.parameters()) + list(graph_decoder.parameters()), lr=0.0004)

# ------------------------------------------------------------------------------# 

current_dir = os.getcwd()
save_dir = os.path.join(current_dir, 'checkpoints', 'sketch_prediction')
os.makedirs(save_dir, exist_ok=True)

def load_models():
    graph_encoder.load_state_dict(torch.load(os.path.join(save_dir, 'graph_encoder.pth')))
    graph_decoder.load_state_dict(torch.load(os.path.join(save_dir, 'graph_decoder.pth')))


def save_models():
    torch.save(graph_encoder.state_dict(), os.path.join(save_dir, 'graph_encoder.pth'))
    torch.save(graph_decoder.state_dict(), os.path.join(save_dir, 'graph_decoder.pth'))


# ------------------------------------------------------------------------------# 

def train():
    # Load the dataset
    dataset = Preprocessing.dataloader.Program_Graph_Dataset('dataset/test')
    print(f"Total number of shape data: {len(dataset)}")
    
    best_val_accuracy = 0
    epochs = 30
    
    graphs = []
    loop_selection_masks = []

    # Preprocess and build the graphs
    for data in tqdm(dataset, desc=f"Building Graphs"):
        # Extract the necessary elements from the dataset
        stroke_cloud_loops, stroke_node_features, connected_stroke_nodes, loop_neighboring_vertical, loop_neighboring_horizontal, loop_neighboring_contained, stroke_to_brep, stroke_operations_order_matrix, final_brep_edges = data
        print("connected_stroke_nodes", connected_stroke_nodes.shape)
        stroke_selection_mask = stroke_operations_order_matrix[:, -1].reshape(-1, 1)
        sketch_selection_mask = stroke_operations_order_matrix[:, -2].reshape(-1, 1)
        extrude_selection_mask = Encoders.helper.choose_extrude_strokes(stroke_selection_mask, sketch_selection_mask, stroke_node_features)

        if not (extrude_selection_mask == 1).any():
            continue

        stroke_node_features = torch.tensor(stroke_node_features, dtype=torch.float32)
        final_brep_edges = torch.tensor(final_brep_edges, dtype=torch.float32)

        gnn_graph = Preprocessing.gnn_graph.SketchLoopGraph(
            stroke_cloud_loops, 
            stroke_node_features, 
            connected_stroke_nodes,
            loop_neighboring_vertical, 
            loop_neighboring_horizontal, 
            loop_neighboring_contained,
            stroke_to_brep
        )
        print("build graph")


#---------------------------------- Public Functions ----------------------------------#


train()