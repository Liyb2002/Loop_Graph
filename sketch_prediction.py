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
graph_decoder = Encoders.gnn.gnn.Sketch_Decoder()

graph_encoder.to(device)
graph_decoder.to(device)

criterion = Encoders.gnn.gnn.FocalLoss(alpha=0.75, gamma=2.5)
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


def compute_accuracy(valid_output, valid_batch_masks):
    batch_size = valid_output.shape[0] // 200
    correct = 0

    for i in range(batch_size):
        output_slice = valid_output[i * 200:(i + 1) * 200]
        mask_slice = valid_batch_masks[i * 200:(i + 1) * 200]

        max_output_value, max_output_index = torch.max(output_slice, dim=0)
        max_mask_value, max_mask_index = torch.max(mask_slice, dim=0)

        values_where_mask_is_1 = output_slice[mask_slice == 1]
        indices_where_mask_is_1 = torch.nonzero(mask_slice == 1).squeeze()

        # print(f"Graph {i}: Values where mask is 1: {values_where_mask_is_1.tolist()}, Indices: {indices_where_mask_is_1.tolist()}")
        # print(f"Graph {i}: max_output_index={max_output_index.item()}, max_mask_index={max_mask_index.item()}")

        if max_output_index == max_mask_index:
            correct += 1

    return correct

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
        stroke_cloud_loops, stroke_node_features, strokes_perpendicular, output_brep_edges, stroke_operations_order_matrix, loop_neighboring_vertical, loop_neighboring_horizontal,loop_neighboring_contained, stroke_to_loop, stroke_to_edge = data

        second_last_column = stroke_operations_order_matrix[:, -2].reshape(-1, 1)
        chosen_strokes = (second_last_column == 1).nonzero(as_tuple=True)[0]  # Indices of chosen strokes
        loop_chosen_mask = []
        for loop in stroke_cloud_loops:
            if all(stroke in chosen_strokes for stroke in loop):
                loop_chosen_mask.append(1)  # Loop is chosen
            else:
                loop_chosen_mask.append(0)  # Loop is not chosen
        
        loop_selection_mask = torch.tensor(loop_chosen_mask, dtype=torch.float).reshape(-1, 1)
        if not (loop_selection_mask == 1).any():
            continue

        # Build the graph
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

        gnn_graph.to_device_withPadding(device)
        loop_selection_mask = loop_selection_mask.to(device)
        print("stroke_node_features", stroke_node_features.shape)
        print("gnn_graph", gnn_graph['stroke'].x.shape)
        # Encoders.helper.vis_stroke_with_order(stroke_node_features)
        Encoders.helper.vis_brep(output_brep_edges)
        # Encoders.helper.vis_selected_loops(gnn_graph, torch.argmax(loop_selection_mask))

        # Prepare the pair
        graphs.append(gnn_graph)
        loop_selection_masks.append(loop_selection_mask)

    print(f"Total number of preprocessed graphs: {len(graphs)}")
    # Split the dataset into training and validation sets (80-20 split)
    split_index = int(0.8 * len(graphs))
    train_graphs, val_graphs = graphs[:split_index], graphs[split_index:]
    train_masks, val_masks = loop_selection_masks[:split_index], loop_selection_masks[split_index:]


    # Convert train and validation graphs to HeteroData
    hetero_train_graphs = [Preprocessing.gnn_graph.convert_to_hetero_data(graph) for graph in train_graphs]
    padded_train_masks = [Preprocessing.dataloader.pad_masks(mask) for mask in train_masks]

    hetero_val_graphs = [Preprocessing.gnn_graph.convert_to_hetero_data(graph) for graph in val_graphs]
    padded_val_masks = [Preprocessing.dataloader.pad_masks(mask) for mask in val_masks]

    # Create DataLoaders for training and validation graphs/masks
    graph_train_loader = DataLoader(hetero_train_graphs, batch_size=16, shuffle=False)
    mask_train_loader = DataLoader(padded_train_masks, batch_size=16, shuffle=False)

    graph_val_loader = DataLoader(hetero_val_graphs, batch_size=16, shuffle=False)
    mask_val_loader = DataLoader(padded_val_masks, batch_size=16, shuffle=False)



    # Training loop
    for epoch in range(epochs):
        train_loss = 0.0
        graph_encoder.train()
        graph_decoder.train()
        train_correct = 0
        train_total = 0


        total_iterations = min(len(graph_train_loader), len(mask_train_loader))
        for hetero_batch, batch_masks in tqdm(zip(graph_train_loader, mask_train_loader), 
                                              desc=f"Epoch {epoch+1}/{epochs} - Training", 
                                              dynamic_ncols=True, 
                                              total=total_iterations):

            optimizer.zero_grad()

            x_dict = graph_encoder(hetero_batch.x_dict, hetero_batch.edge_index_dict)
            output = graph_decoder(x_dict)

            batch_masks = batch_masks.to(output.device).view(-1, 1)
            valid_mask = (batch_masks != -1).float()
            valid_output = output * valid_mask
            valid_batch_masks = batch_masks * valid_mask
            loss = criterion(valid_output, valid_batch_masks)

            train_correct += compute_accuracy(valid_output, valid_batch_masks)
            train_total += 16


            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_accuracy = train_correct / train_total if train_total > 0 else 0
        train_loss /= len(train_graphs)
        val_loss = 0.0
        correct = 0
        total = 0

        graph_encoder.eval()
        graph_decoder.eval()
        with torch.no_grad():
            total_iterations_val = min(len(graph_val_loader), len(mask_val_loader))

            for hetero_batch, batch_masks in tqdm(zip(graph_val_loader, mask_val_loader), 
                                                  desc="Validation", 
                                                  dynamic_ncols=True, 
                                                  total=total_iterations_val):
                x_dict = graph_encoder(hetero_batch.x_dict, hetero_batch.edge_index_dict)
                output = graph_decoder(x_dict)

                batch_masks = batch_masks.to(output.device).view(-1, 1)
                valid_mask = (batch_masks != -1).float()
                valid_output = output * valid_mask
                valid_batch_masks = batch_masks * valid_mask
                loss = criterion(valid_output, valid_batch_masks)

                correct += compute_accuracy(valid_output, valid_batch_masks)
                total += 16

        
        val_loss /= len(val_graphs)
        

        accuracy = correct / total if total > 0 else 0
        print(f"Epoch {epoch+1}/{epochs} - Training Loss: {train_loss:.5f} - Validation Loss: {val_loss:.5f} - Train Accuracy: {train_accuracy:.5f} - Validation Accuracy: {accuracy:.5f}")

        if accuracy > best_val_accuracy:
            best_val_accuracy = accuracy
            save_models()
            print(f"Models saved at epoch {epoch+1} with validation accuracy: {accuracy:.5f}")



def eval():
    load_models()
    # Load the dataset
    dataset = Preprocessing.dataloader.Program_Graph_Dataset('dataset/messy_order')
    print(f"Total number of shape data: {len(dataset)}")


    graphs = []
    loop_selection_masks = []

    # Preprocess and build the graphs
    for data in tqdm(dataset, desc=f"Building Graphs"):
        # Extract the necessary elements from the dataset
        stroke_cloud_loops, stroke_node_features, strokes_perpendicular, loop_neighboring_vertical, loop_neighboring_horizontal, loop_neighboring_contained, loop_neighboring_coplanar, stroke_to_loop, stroke_to_edge ,stroke_operations_order_matrix = data

        second_last_column = stroke_operations_order_matrix[:, -2].reshape(-1, 1)
        chosen_strokes = (second_last_column == 1).nonzero(as_tuple=True)[0]  # Indices of chosen strokes
        loop_chosen_mask = []
        for loop in stroke_cloud_loops:
            if all(stroke in chosen_strokes for stroke in loop):
                loop_chosen_mask.append(1)  # Loop is chosen
            else:
                loop_chosen_mask.append(0)  # Loop is not chosen
        
        loop_selection_mask = torch.tensor(loop_chosen_mask, dtype=torch.float).reshape(-1, 1)
        if not (loop_selection_mask == 1).any():
            continue

        # Build the graph
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

        # Encoders.helper.vis_brep(final_brep_edges)

        # Prepare the pair
        graphs.append(gnn_graph)
        loop_selection_masks.append(loop_selection_mask)


    # Eval
    graph_encoder.eval()
    graph_decoder.eval()
    lv1_correct = 0
    lv1_total = 0
    lv2_correct = 0
    lv2_total = 0
    lv3_correct = 0
    lv3_total = 0
    lv4_correct = 0
    lv4_total = 0

    eval_loss = 0.0

    with torch.no_grad():
        for gnn_graph, loop_selection_mask in tqdm(zip(graphs, loop_selection_masks), desc=f"Evaluation"):
            x_dict = graph_encoder(gnn_graph.x_dict, gnn_graph.edge_index_dict)
            output = graph_decoder(x_dict)
            
            # Encoders.helper.vis_selected_loops(gnn_graph, torch.argmax(output))
            # Encoders.helper.vis_selected_loops(gnn_graph, torch.argmax(loop_selection_mask))


            if x_dict['loop'].shape[0] < 15: 
                lv1_total += 1
            elif x_dict['loop'].shape[0] < 40: 
                lv2_total += 1
            elif x_dict['loop'].shape[0] < 60: 
                lv3_total += 1
            elif x_dict['loop'].shape[0] < 200: 
                lv4_total += 1

            # Check if the selected loop is correct
            if torch.argmax(output) == torch.argmax(loop_selection_mask):
                if x_dict['loop'].shape[0] < 15: 
                    lv1_correct += 1
                elif x_dict['loop'].shape[0] < 40: 
                    lv2_correct += 1
                elif x_dict['loop'].shape[0] < 60: 
                    lv3_correct += 1
                elif x_dict['loop'].shape[0] < 200: 
                    lv4_correct += 1

            # else:
                # Encoders.helper.vis_whole_graph(gnn_graph, torch.argmax(output))
                # Encoders.helper.vis_whole_graph(gnn_graph, torch.argmax(loop_selection_mask))


            loss = criterion(output, loop_selection_mask)
            eval_loss += loss.item()
        
        eval_loss /= len(graphs)

    lv1_accuracy = lv1_correct / lv1_total if lv1_total > 0 else 0 
    lv2_accuracy = lv2_correct / lv2_total if lv2_total > 0 else 0 
    lv3_accuracy = lv3_correct / lv3_total if lv3_total > 0 else 0 
    lv4_accuracy = lv4_correct / lv4_total if lv4_total > 0 else 0 


    print(f"lv1_accuracy: {lv1_accuracy:.5f}")
    print(f"lv2_accuracy: {lv2_accuracy:.5f}")
    print(f"lv3_accuracy: {lv3_accuracy:.5f}")
    print(f"lv4_accuracy: {lv4_accuracy:.5f}")



#---------------------------------- Public Functions ----------------------------------#


train()