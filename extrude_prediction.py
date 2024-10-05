import Preprocessing.dataloader
import Preprocessing.gnn_graph
import Preprocessing.gnn_graph_stroke

import Encoders.gnn.gnn
import Encoders.helper

import Preprocessing.proc_CAD
import Preprocessing.proc_CAD.helper
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

def train():
    # Load the dataset
    dataset = Preprocessing.dataloader.Program_Graph_Dataset('dataset/messy_order')
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

        if len(graphs) > 50:
            break
        # Encoders.helper.vis_stroke_graph(gnn_graph, extrude_selection_mask)



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


        for gnn_graph, sketch_selection_mask in tqdm(zip(train_graphs, train_masks), desc=f"Epoch {epoch+1}/{epochs} - Training", dynamic_ncols=True):
            optimizer.zero_grad()

            x_dict = graph_encoder(gnn_graph.x_dict, gnn_graph.edge_index_dict)
            output = graph_decoder(x_dict)

            
            # Encoders.helper.vis_stroke_graph(gnn_graph, sketch_selection_mask)
            # Encoders.helper.vis_stroke_graph(gnn_graph, output.detach())


            loss = criterion(output, sketch_selection_mask)
            
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

                if torch.all(condition_1 | condition_2):
                    correct += 1
                else:
                    pass

                total += 1
                val_loss += loss.item()
        

        train_loss /= len(train_graphs)
        val_loss /= len(val_graphs)
        

        accuracy = correct / total if total > 0 else 0
        print(f"Epoch {epoch+1}/{epochs} - Training Loss: {train_loss:.5f} - Validation Loss: {val_loss:.5f} - Validation Accuracy: {accuracy:.5f}")

        if best_accuracy <  accuracy:
            best_accuracy = accuracy
            save_models()
            print(f"Models saved at epoch {epoch+1} with validation accuracy: {accuracy:.5f}")



def eval():
    load_models()
    # Load the dataset
    dataset = Preprocessing.dataloader.Program_Graph_Dataset('dataset/messy_order_eval')
    print(f"Total number of shape data: {len(dataset)}")
        
    graphs = []
    stroke_selection_masks = []

    lv1_correct = 0
    lv1_total = 0
    lv2_correct = 0
    lv2_total = 0
    lv3_correct = 0
    lv3_total = 0
    lv4_correct = 0
    lv4_total = 0

    correct = 0
    total = 0

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

        if not (extrude_selection_mask == 1).any() and not (sketch_selection_mask == 1).any():
            continue

        stroke_node_features = torch.tensor(stroke_node_features, dtype=torch.float32)
        final_brep_edges = torch.tensor(final_brep_edges, dtype=torch.float32)


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

        graphs.append(gnn_graph)
        stroke_selection_masks.append(extrude_selection_mask)

    with torch.no_grad():
        for gnn_graph, loop_selection_mask in tqdm(zip(graphs, stroke_selection_masks), desc=f"Evaluation"):
            x_dict = graph_encoder(gnn_graph.x_dict, gnn_graph.edge_index_dict)
            output = graph_decoder(x_dict)
            

            condition_1 = (loop_selection_mask == 1) & (output > 0.5)
            condition_2 = (loop_selection_mask == 0) & (output < 0.5)

            if torch.all(condition_1 | condition_2):
                correct += 1
                if x_dict['loop'].shape[0] < 15: 
                    lv1_correct += 1
                elif x_dict['loop'].shape[0] < 35: 
                    lv2_correct += 1
                elif x_dict['loop'].shape[0] < 50: 
                    lv3_correct += 1
                elif x_dict['loop'].shape[0] < 200: 
                    lv4_correct += 1

            else:
                indices = torch.nonzero(loop_selection_mask == 1, as_tuple=True)[0]
                # output_values_at_indices = output[indices]
                # print("Output values where loop_selection_mask is 1", output_values_at_indices)

                Encoders.helper.vis_stroke_graph(gnn_graph, loop_selection_mask)
                Encoders.helper.vis_stroke_graph(gnn_graph, output.detach())

            if x_dict['loop'].shape[0] < 15: 
                lv1_total += 1
            elif x_dict['loop'].shape[0] < 35: 
                lv2_total += 1
            elif x_dict['loop'].shape[0] < 50: 
                lv3_total += 1
            elif x_dict['loop'].shape[0] < 200: 
                lv4_total += 1
            total += 1


    lv1_accuracy = lv1_correct / lv1_total if lv1_total > 0 else 0 
    lv2_accuracy = lv2_correct / lv2_total if lv2_total > 0 else 0 
    lv3_accuracy = lv3_correct / lv3_total if lv3_total > 0 else 0 
    lv4_accuracy = lv4_correct / lv4_total if lv4_total > 0 else 0 
    accuracy = correct / total if total > 0 else 0 


    print(f"lv1_accuracy: {lv1_accuracy:.5f}")
    print(f"lv2_accuracy: {lv2_accuracy:.5f}")
    print(f"lv3_accuracy: {lv3_accuracy:.5f}")
    print(f"lv4_accuracy: {lv4_accuracy:.5f}")
    print(f"accuracy: {accuracy:.5f}")


#---------------------------------- Public Functions ----------------------------------#


train()