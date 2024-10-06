import Preprocessing.dataloader
import Preprocessing.gnn_graph
import Preprocessing.gnn_graph_stroke

import Encoders.gnn.gnn
import Encoders.helper

import Preprocessing.proc_CAD
import Preprocessing.proc_CAD.helper
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch

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
graph_decoder = Encoders.gnn.gnn.Extrude_Decoder()

graph_encoder.to(device)
graph_decoder.to(device)

criterion = nn.BCELoss()
optimizer = optim.Adam(list(graph_encoder.parameters()) + list(graph_decoder.parameters()), lr=0.0004)

# ------------------------------------------------------------------------------# 

current_dir = os.getcwd()
save_dir = os.path.join(current_dir, 'checkpoints', 'extrude_prediction')
os.makedirs(save_dir, exist_ok=True)

def load_models():
    graph_encoder.load_state_dict(torch.load(os.path.join(save_dir, 'graph_encoder.pth')))
    graph_decoder.load_state_dict(torch.load(os.path.join(save_dir, 'graph_decoder.pth')))


def save_models():
    torch.save(graph_encoder.state_dict(), os.path.join(save_dir, 'graph_encoder.pth'))
    torch.save(graph_decoder.state_dict(), os.path.join(save_dir, 'graph_decoder.pth'))


# ------------------------------------------------------------------------------# 


def compute_accuracy(output, loop_selection_mask):
    condition_1 = (loop_selection_mask == 1) & (output > 0.5)
    condition_2 = (loop_selection_mask == 0) & (output < 0.5)
    if torch.all(condition_1 | condition_2):
        return 1
    else:
        return 0


# ------------------------------------------------------------------------------# 



def train():
    # Load the dataset
    dataset = Preprocessing.dataloader.Program_Graph_Dataset('dataset/messy_order_full')
    print(f"Total number of shape data: {len(dataset)}")
    
    best_accuracy = 0
    epochs = 100
    
    graphs = []
    stroke_selection_masks = []

    # Preprocess and build the graphs
    for data in tqdm(dataset, desc=f"Building Graphs"):
        # Extract the necessary elements from the dataset
        stroke_cloud_loops, stroke_node_features, strokes_perpendicular, loop_neighboring_vertical, loop_neighboring_horizontal, loop_neighboring_contained, loop_neighboring_coplanar, stroke_to_loop, stroke_to_edge ,stroke_operations_order_matrix = data

        stroke_selection_mask = stroke_operations_order_matrix[:, -1].reshape(-1, 1)
        sketch_selection_mask = stroke_operations_order_matrix[:, -2].reshape(-1, 1)
        extrude_selection_mask = Encoders.helper.choose_extrude_strokes(stroke_selection_mask, sketch_selection_mask, stroke_node_features)

        # find the sketch_loops
        chosen_strokes = (sketch_selection_mask == 1).nonzero(as_tuple=True)[0]  # Indices of chosen strokes
        loop_chosen_mask = []
        for loop in stroke_cloud_loops:
            if all(stroke in chosen_strokes for stroke in loop):
                loop_chosen_mask.append(1)  # Loop is chosen
            else:
                loop_chosen_mask.append(0)  # Loop is not chosen
        
        sketch_loop_selection_mask = torch.tensor(loop_chosen_mask, dtype=torch.float).reshape(-1, 1)


        if not (extrude_selection_mask == 1).any() and not (sketch_loop_selection_mask == 1).any():
            continue


        stroke_node_features = torch.tensor(stroke_node_features, dtype=torch.float32)

        gnn_graph = Preprocessing.gnn_graph.SketchLoopGraph(
            stroke_cloud_loops, 
            stroke_node_features, 
            strokes_perpendicular, 
            loop_neighboring_vertical, 
            loop_neighboring_horizontal, 
            loop_neighboring_contained,
            stroke_to_loop,
            stroke_to_edge
        )

        gnn_graph.set_select_sketch(sketch_loop_selection_mask)
        gnn_graph.to_device(device)
        extrude_selection_mask = extrude_selection_mask.to(device)


        graphs.append(gnn_graph)
        stroke_selection_masks.append(extrude_selection_mask)

        # Encoders.helper.vis_stroke_graph(gnn_graph, extrude_selection_mask)



    print(f"Total number of preprocessed graphs: {len(graphs)}")
    # Split the dataset into training and validation sets (80-20 split)
    split_index = int(0.8 * len(graphs))
    train_graphs, val_graphs = graphs[:split_index], graphs[split_index:]
    train_masks, val_masks = stroke_selection_masks[:split_index], stroke_selection_masks[split_index:]


    # Convert train and validation graphs to HeteroData
    hetero_train_graphs = [Preprocessing.gnn_graph.convert_to_hetero_data(graph) for graph in train_graphs]
    padded_train_masks = [Preprocessing.dataloader.pad_masks(mask) for mask in train_masks]

    hetero_val_graphs = [Preprocessing.gnn_graph.convert_to_hetero_data(graph) for graph in val_graphs]
    padded_val_masks = [Preprocessing.dataloader.pad_masks(mask) for mask in val_masks]

    # Create DataLoaders for training and validation graphs/masks
    graph_train_loader = DataLoader(hetero_train_graphs, batch_size=16, shuffle=True)
    mask_train_loader = DataLoader(padded_train_masks, batch_size=16, shuffle=True)

    graph_val_loader = DataLoader(hetero_val_graphs, batch_size=16, shuffle=False)
    mask_val_loader = DataLoader(padded_val_masks, batch_size=16, shuffle=False)

    # Training and validation loop
    epochs = 10  # Number of epochs
    best_accuracy = 0.0

    for epoch in range(epochs):
        train_loss = 0.0
        correct = 0
        total = 0

        graph_encoder.train()
        graph_decoder.train()

        # Training loop
        for hetero_batch, batch_masks in tqdm(zip(graph_train_loader, mask_train_loader), desc=f"Epoch {epoch+1}/{epochs} - Training", dynamic_ncols=True):

            optimizer.zero_grad()

            # Forward pass through the graph encoder
            x_dict = graph_encoder(hetero_batch.x_dict, hetero_batch.edge_index_dict)

            # Forward pass through the graph decoder
            output = graph_decoder(x_dict)

            # Ensure masks are on the correct device and reshape them to match the output
            batch_masks = batch_masks.to(output.device).view(-1, 1)

            # Compute the loss
            loss = criterion(output, batch_masks)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            # Accuracy computation using the preferred method
            correct += compute_accuracy(output, batch_masks)
            total += batch_masks.size(0) // 200  # Since each graph has 200 loops

        train_accuracy = correct / total
        print(f"Epoch {epoch+1}/{epochs}, Training Loss: {train_loss / len(graph_train_loader)}, Training Accuracy: {train_accuracy:.4f}")

        # Validation loop
        val_loss = 0.0
        correct = 0
        total = 0
        graph_encoder.eval()
        graph_decoder.eval()

        with torch.no_grad():
            for hetero_batch, batch_masks in tqdm(zip(graph_val_loader, mask_val_loader), desc="Validation", dynamic_ncols=True):
                # Forward pass through the graph encoder
                x_dict = graph_encoder(hetero_batch.x_dict, hetero_batch.edge_index_dict)

                # Forward pass through the graph decoder
                output = graph_decoder(x_dict)

                # Ensure masks are on the correct device and reshape them
                batch_masks = batch_masks.to(output.device).view(-1, 1)

                # Compute the validation loss
                loss = criterion(output, batch_masks)
                val_loss += loss.item()

                # Accuracy computation using the preferred method
                correct += compute_accuracy(output, batch_masks)
                total += batch_masks.size(0) // 200

        val_accuracy = correct / total
        print(f"Validation Loss: {val_loss / len(graph_val_loader)}, Validation Accuracy: {val_accuracy:.4f}")

        # Save the model if validation accuracy improves
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            print(f"New best accuracy: {best_accuracy:.4f}, saving model...")
            save_models(graph_encoder, graph_decoder)  # Call your save_models function here


def eval():
    # Load the saved models
    load_models()
    graph_encoder.eval()
    graph_decoder.eval()

    # Load the dataset
    dataset = Preprocessing.dataloader.Program_Graph_Dataset('dataset/messy_order')
    print(f"Total number of shape data: {len(dataset)}")

    graphs = []
    stroke_selection_masks = []

    # Preprocess and build the graphs (same as in training)
    for data in tqdm(dataset, desc=f"Building Graphs"):
        stroke_cloud_loops, stroke_node_features, strokes_perpendicular, loop_neighboring_vertical, loop_neighboring_horizontal, loop_neighboring_contained, loop_neighboring_coplanar, stroke_to_loop, stroke_to_edge ,stroke_operations_order_matrix = data

        stroke_selection_mask = stroke_operations_order_matrix[:, -1].reshape(-1, 1)
        sketch_selection_mask = stroke_operations_order_matrix[:, -2].reshape(-1, 1)
        extrude_selection_mask = Encoders.helper.choose_extrude_strokes(stroke_selection_mask, sketch_selection_mask, stroke_node_features)

        chosen_strokes = (sketch_selection_mask == 1).nonzero(as_tuple=True)[0]
        loop_chosen_mask = []
        for loop in stroke_cloud_loops:
            if all(stroke in chosen_strokes for stroke in loop):
                loop_chosen_mask.append(1)
            else:
                loop_chosen_mask.append(0)
        
        sketch_loop_selection_mask = torch.tensor(loop_chosen_mask, dtype=torch.float).reshape(-1, 1)

        if not (extrude_selection_mask == 1).any() and not (sketch_loop_selection_mask == 1).any():
            continue

        stroke_node_features = torch.tensor(stroke_node_features, dtype=torch.float32)

        gnn_graph = Preprocessing.gnn_graph.SketchLoopGraph(
            stroke_cloud_loops, 
            stroke_node_features, 
            strokes_perpendicular, 
            loop_neighboring_vertical, 
            loop_neighboring_horizontal, 
            loop_neighboring_contained,
            stroke_to_loop,
            stroke_to_edge
        )

        gnn_graph.set_select_sketch(sketch_loop_selection_mask)
        gnn_graph.to_device(device)
        extrude_selection_mask = extrude_selection_mask.to(device)

        graphs.append(gnn_graph)
        stroke_selection_masks.append(extrude_selection_mask)

    print(f"Total number of preprocessed graphs: {len(graphs)}")

    # Convert to HeteroData and pad the masks
    hetero_graphs = [Preprocessing.gnn_graph.convert_to_hetero_data(graph) for graph in graphs]
    padded_masks = [Preprocessing.dataloader.pad_masks(mask) for mask in stroke_selection_masks]

    # Create DataLoader for evaluation
    graph_eval_loader = DataLoader(hetero_graphs, batch_size=16, shuffle=False)
    mask_eval_loader = DataLoader(padded_masks, batch_size=16, shuffle=False)

    val_loss = 0.0
    correct = 0
    total = 0

    criterion = torch.nn.BCELoss()  # Assuming BCELoss is used in the training

    with torch.no_grad():
        for hetero_batch, batch_masks in tqdm(zip(graph_eval_loader, mask_eval_loader), desc="Evaluation", dynamic_ncols=True):
            # Forward pass through the graph encoder
            x_dict = graph_encoder(hetero_batch.x_dict, hetero_batch.edge_index_dict)

            # Forward pass through the graph decoder
            output = graph_decoder(x_dict)

            # Ensure masks are on the correct device and reshape them
            batch_masks = batch_masks.to(output.device).view(-1, 1)

            # Compute the loss
            loss = criterion(output, batch_masks)
            val_loss += loss.item()

            # Accuracy computation using your custom method
            print("output", output[:20])
            print("batch_masks", batch_masks[:20])
            correct += compute_accuracy(output, batch_masks)
            total += batch_masks.size(0) // 200

    val_accuracy = correct / total
    print(f"Evaluation Loss: {val_loss / len(graph_eval_loader)}, Evaluation Accuracy: {val_accuracy:.4f}")




#---------------------------------- Public Functions ----------------------------------#


eval()