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

graph_encoder = Encoders.gnn.gnn.SemanticModule()
graph_decoder = Encoders.gnn.gnn.Sketch_prediction()

graph_encoder.to(device)
graph_decoder.to(device)

criterion = nn.BCELoss()
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
    dataset = Preprocessing.dataloader.Program_Graph_Dataset('dataset/simple')
    print(f"Total number of shape data: {len(dataset)}")
    
    best_val_loss = float('inf')
    epochs = 30
    
    graphs = []
    loop_selection_masks = []

    # Preprocess and build the graphs
    for data in tqdm(dataset, desc=f"Building Graphs"):
        # Extract the necessary elements from the dataset
        stroke_cloud_loops, stroke_node_features, loop_neighboring_vertical, loop_neighboring_horizontal, loop_neighboring_contained, stroke_to_brep, stroke_operations_order_matrix, final_brep_edges = data

        second_last_column = stroke_operations_order_matrix[:, -2].reshape(-1, 1)
        chosen_strokes = (second_last_column == 1).nonzero(as_tuple=True)[0]  # Indices of chosen strokes
        loop_chosen_mask = []
        for loop in stroke_cloud_loops:
            if all(stroke in chosen_strokes for stroke in loop):
                loop_chosen_mask.append(1)  # Loop is chosen
            else:
                loop_chosen_mask.append(0)  # Loop is not chosen
        
        loop_selection_mask = torch.tensor(loop_chosen_mask, dtype=torch.float).reshape(-1, 1).to(device)
        if not (loop_selection_mask == 1).any():
            continue

        # Build the graph
        gnn_graph = Preprocessing.gnn_graph.SketchLoopGraph(
            stroke_cloud_loops, 
            stroke_node_features, 
            loop_neighboring_vertical, 
            loop_neighboring_horizontal, 
            loop_neighboring_contained,
            stroke_to_brep
        )

        # Encoders.helper.vis_brep(final_brep_edges)
        # Encoders.helper.vis_whole_graph(gnn_graph, torch.argmax(loop_selection_mask))

        # Prepare the pair
        graphs.append(gnn_graph)
        loop_selection_masks.append(loop_selection_mask)

    print(f"Total number of preprocessed graphs: {len(graphs)}")
    # Split the dataset into training and validation sets (80-20 split)
    split_index = int(0.8 * len(graphs))
    train_graphs, val_graphs = graphs[:split_index], graphs[split_index:]
    train_masks, val_masks = loop_selection_masks[:split_index], loop_selection_masks[split_index:]




    # Training loop
    for epoch in range(epochs):
        train_loss = 0.0
        graph_encoder.train()
        graph_decoder.train()

        for gnn_graph, loop_selection_mask in tqdm(zip(train_graphs, train_masks), desc=f"Epoch {epoch+1}/{epochs} - Training"):

            optimizer.zero_grad()

            x_dict = graph_encoder(gnn_graph.x_dict, gnn_graph.edge_index_dict)
            output = graph_decoder(x_dict)

            loss = criterion(output, loop_selection_mask)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_graphs)
        val_loss = 0.0
        min_val_loss = float('inf')
        graph_encoder.eval()
        graph_decoder.eval()
        with torch.no_grad():
            for gnn_graph, loop_selection_mask in tqdm(zip(val_graphs, val_masks), desc=f"Epoch {epoch+1}/{epochs} - Validation"):
                x_dict = graph_encoder(gnn_graph.x_dict, gnn_graph.edge_index_dict)
                output = graph_decoder(x_dict)

                loss = criterion(output, loop_selection_mask)
                val_loss += loss.item()
        
        val_loss /= len(val_graphs)

        print(f"Epoch {epoch+1}/{epochs} - Training Loss: {train_loss:.4f} - Validation Loss: {val_loss:.4f}")

        if val_loss < min_val_loss:
            min_val_loss = val_loss
            save_models()
            print(f"Models saved at epoch {epoch+1} with validation loss: {val_loss:.4f}")



def eval():
    load_models()
    # Load the dataset
    dataset = Preprocessing.dataloader.Program_Graph_Dataset('dataset/eval')
    print(f"Total number of shape data: {len(dataset)}")


    graphs = []
    loop_selection_masks = []

    # Preprocess and build the graphs
    for data in tqdm(dataset, desc=f"Building Graphs"):
        # Extract the necessary elements from the dataset
        stroke_cloud_loops, stroke_node_features, loop_neighboring_vertical, loop_neighboring_horizontal, loop_neighboring_contained, stroke_to_brep, stroke_operations_order_matrix, final_brep_edges = data

        second_last_column = stroke_operations_order_matrix[:, -2].reshape(-1, 1)
        chosen_strokes = (second_last_column == 1).nonzero(as_tuple=True)[0]  # Indices of chosen strokes
        loop_chosen_mask = []
        for loop in stroke_cloud_loops:
            if all(stroke in chosen_strokes for stroke in loop):
                loop_chosen_mask.append(1)  # Loop is chosen
            else:
                loop_chosen_mask.append(0)  # Loop is not chosen
        
        loop_selection_mask = torch.tensor(loop_chosen_mask, dtype=torch.float).reshape(-1, 1).to(device)
        if not (loop_selection_mask == 1).any():
            continue

        # Build the graph
        gnn_graph = Preprocessing.gnn_graph.SketchLoopGraph(
            stroke_cloud_loops, 
            stroke_node_features, 
            loop_neighboring_vertical, 
            loop_neighboring_horizontal, 
            loop_neighboring_contained,
            stroke_to_brep
        )

        # Encoders.helper.vis_brep(final_brep_edges)
        # Encoders.helper.vis_whole_graph(gnn_graph, torch.argmax(loop_selection_mask))

        # Prepare the pair
        graphs.append(gnn_graph)
        loop_selection_masks.append(loop_selection_mask)


    # Eval
    graph_encoder.eval()
    graph_decoder.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for gnn_graph, loop_selection_mask in tqdm(zip(graphs, loop_selection_masks), desc=f"Evaluation"):
            x_dict = graph_encoder(gnn_graph.x_dict, gnn_graph.edge_index_dict)
            output = graph_decoder(x_dict)

            output_selected_idx = torch.argmax(output)  # Index of selected loop in output
            mask_selected_idx = torch.argmax(loop_selection_mask)  # Index of selected loop in ground truth
            
            # Check if the selected loop is correct
            if output_selected_idx == mask_selected_idx:
                correct += 1
            # else:
            #     Encoders.helper.vis_whole_graph(gnn_graph, torch.argmax(output))
            #     Encoders.helper.vis_whole_graph(gnn_graph, torch.argmax(loop_selection_mask))
            total += 1

    accuracy = correct / total if total > 0 else 0
    print(f"Validation Accuracy: {accuracy:.4f}")

#---------------------------------- Public Functions ----------------------------------#


eval()