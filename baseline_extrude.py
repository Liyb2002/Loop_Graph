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


graph_encoder = Encoders.gnn.gnn.Sketch_base_SemanticModule()
graph_decoder = Encoders.gnn.gnn.Extrude_prediction()

graph_encoder.to(device)
graph_decoder.to(device)

criterion = nn.BCELoss()
optimizer = optim.Adam(list(graph_encoder.parameters()) + list(graph_decoder.parameters()), lr=0.001)


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


def train_extrude_prediction_baseline():
    # Load the dataset
    dataset = Preprocessing.dataloader.Program_Graph_Dataset('dataset/simple')
    print(f"Total number of shape data: {len(dataset)}")

    best_val_loss = 100.0
    epochs = 10

    graphs = []
    stroke_selection_masks = []

    # Preprocess and build the graphs
    for data in tqdm(dataset, desc=f"Building Graphs"):
        # Extract the necessary elements from the dataset
        stroke_cloud_loops, stroke_node_features, loop_neighboring_vertical, loop_neighboring_horizontal, loop_neighboring_contained, stroke_to_brep, stroke_operations_order_matrix, final_brep_edges = data

        stroke_selection_mask = stroke_operations_order_matrix[:, -1].reshape(-1, 1)
        sketch_selection_mask = stroke_operations_order_matrix[:, -2].reshape(-1, 1)
        extrude_selection_mask = Encoders.helper.choose_extrude_strokes(stroke_selection_mask, sketch_selection_mask, stroke_node_features)

        if not (extrude_selection_mask == 1).any():
            continue
        
        
        stroke_node_features = torch.tensor(stroke_node_features, dtype=torch.float32)
        final_brep_edges = torch.tensor(final_brep_edges, dtype=torch.float32)

        # Build the graph
        gnn_graph = Preprocessing.gnn_graph.SketchHeteroData(
            stroke_node_features, 
            final_brep_edges
        )

        # Encoders.helper.vis_brep(final_brep_edges)
        # Encoders.helper.vis_stroke_graph(gnn_graph, extrude_selection_mask)

        # Prepare the pair
        graphs.append(gnn_graph)
        stroke_selection_masks.append(extrude_selection_mask)


    print(f"Total number of preprocessed graphs: {len(graphs)}")
    # Split the dataset into training and validation sets (80-20 split)
    split_index = int(0.8 * len(graphs))
    train_graphs, val_graphs = graphs[:split_index], graphs[split_index:]
    train_masks, val_masks = stroke_selection_masks[:split_index], stroke_selection_masks[split_index:]



    for epoch in range(epochs):
        train_loss = 0.0
        graph_encoder.train()
        graph_decoder.train()

        val_loss = 0.0
        correct = 0
        total = 0


        for gnn_graph, loop_selection_mask in tqdm(zip(train_graphs, train_masks), desc=f"Epoch {epoch+1}/{epochs} - Training", dynamic_ncols=True):

            optimizer.zero_grad()

            x_dict = graph_encoder(gnn_graph.x_dict, gnn_graph.edge_index_dict)
            output = graph_decoder(x_dict)
            
            loss = criterion(output, loop_selection_mask)
            
            loss.backward()
            optimizer.step()

            train_loss += loss.item()


        graph_encoder.eval()
        graph_decoder.eval()
        
        with torch.no_grad():
            for gnn_graph, loop_selection_mask in tqdm(zip(val_graphs, val_masks), desc=f"Epoch {epoch+1}/{epochs} - Validation"):
                x_dict = graph_encoder(gnn_graph.x_dict, gnn_graph.edge_index_dict)
                output = graph_decoder(x_dict)
                loss = criterion(output, loop_selection_mask)


                condition_1 = (loop_selection_mask == 1) & (output > 0.5)
                condition_2 = (loop_selection_mask == 0) & (output < 0.5)

                # chosen_index_mask = torch.where(loop_selection_mask > 0.5)[0]
                # print(f"Chosen indices in loop_selection_mask: {chosen_index_mask.tolist()}")
                # chosen_index_output = torch.where(output > 0.5)[0]
                # print(f"Chosen indices in output: {chosen_index_output.tolist()}")

                if torch.all(condition_1 | condition_2):
                    correct += 1
                else:
                    pass
                    # if epoch > 28:
                    #     print("output", output)
                    #     Encoders.helper.vis_stroke_graph(gnn_graph, loop_selection_mask)
                    #     Encoders.helper.vis_stroke_graph(gnn_graph, output)

                total += 1
                val_loss += loss.item()
        

        train_loss /= len(train_graphs)
        val_loss /= len(val_graphs)
        

        accuracy = correct / total if total > 0 else 0
        print(f"Epoch {epoch+1}/{epochs} - Training Loss: {train_loss:.5f} - Validation Loss: {val_loss:.5f} - Validation Accuracy: {accuracy:.5f}")

        if val_loss <  best_val_loss:
            best_val_loss = val_loss
            save_models()
            print(f"Models saved at epoch {epoch+1} with validation accuracy: {accuracy:.5f}")



def eval_extrude_prediction_baseline():
    load_models()
    dataset = Preprocessing.dataloader.Program_Graph_Dataset('dataset/simple')


    graphs = []
    stroke_selection_masks = []
    correct = 0
    total = 0

    for data in tqdm(dataset, desc=f"Building Graphs"):
        # Extract the necessary elements from the dataset
        stroke_cloud_loops, stroke_node_features, loop_neighboring_vertical, loop_neighboring_horizontal, loop_neighboring_contained, stroke_to_brep, stroke_operations_order_matrix, final_brep_edges = data

        stroke_selection_mask = stroke_operations_order_matrix[:, -1].reshape(-1, 1)
        sketch_selection_mask = stroke_operations_order_matrix[:, -2].reshape(-1, 1)
        extrude_selection_mask = Encoders.helper.choose_extrude_strokes(stroke_selection_mask, sketch_selection_mask, stroke_node_features)

        if not (extrude_selection_mask == 1).any():
            continue
        
        
        stroke_node_features = torch.tensor(stroke_node_features, dtype=torch.float32)
        final_brep_edges = torch.tensor(final_brep_edges, dtype=torch.float32)

        # Build the graph
        gnn_graph = Preprocessing.gnn_graph.SketchHeteroData(
            stroke_node_features, 
            final_brep_edges
        )

        # Encoders.helper.vis_brep(final_brep_edges)
        # Encoders.helper.vis_stroke_graph(gnn_graph, sketch_selection_mask)
        # Encoders.helper.vis_stroke_graph(gnn_graph, extrude_selection_mask)

        # Prepare the pair
        graphs.append(gnn_graph)
        stroke_selection_masks.append(extrude_selection_mask)

    
    for gnn_graph, loop_selection_mask in tqdm(zip(graphs, stroke_selection_masks), dynamic_ncols=True):
        graph_encoder.eval()
        graph_decoder.eval()
        x_dict = graph_encoder(gnn_graph.x_dict, gnn_graph.edge_index_dict)
        output = graph_decoder(x_dict)


        condition_1 = (loop_selection_mask == 1) & (output > 0.5)
        condition_2 = (loop_selection_mask == 0) & (output < 0.5)

        if torch.all(condition_1 | condition_2):
            correct += 1
        else:
            Encoders.helper.vis_stroke_graph(gnn_graph, output.detach())
            Encoders.helper.vis_stroke_graph(gnn_graph, loop_selection_mask)
        total += 1


        accuracy = correct / total if total > 0 else 0

    print(f"Accuracy: {accuracy:.5f}")

#---------------------------------- Public Functions ----------------------------------#


train_extrude_prediction_baseline()