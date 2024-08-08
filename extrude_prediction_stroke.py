import Preprocessing.dataloader
import Preprocessing.gnn_graph_stroke
import Preprocessing.SBGCN.SBGCN_graph
import Preprocessing.SBGCN.SBGCN_network

import Models.sketch_model_helper
import Encoders.gnn_stroke.gnn
import Encoders.helper

from torch.utils.data import DataLoader, random_split, Subset
from tqdm import tqdm
from Preprocessing.config import device
import torch
import torch.nn as nn
import torch.optim as optim
import os
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


graph_encoder = Encoders.gnn_stroke.gnn.SemanticModule()
graph_decoder = Encoders.gnn_stroke.gnn.ExtrudingStrokePrediction()

current_dir = os.getcwd()
save_dir = os.path.join(current_dir, 'checkpoints', 'extrude_stroke')
os.makedirs(save_dir, exist_ok=True)

def save_models():
    torch.save(graph_encoder.state_dict(), os.path.join(save_dir, 'graph_encoder.pth'))
    torch.save(graph_decoder.state_dict(), os.path.join(save_dir, 'graph_decoder.pth'))


def load_models():
    graph_encoder.load_state_dict(torch.load(os.path.join(save_dir, 'graph_encoder.pth')))
    graph_decoder.load_state_dict(torch.load(os.path.join(save_dir, 'graph_decoder.pth')))


# Define optimizer and loss function
optimizer = optim.Adam( list(graph_encoder.parameters()) + list(graph_decoder.parameters()), lr=0.0005)
loss_function = nn.BCELoss()


def train():
    # Load the dataset
    dataset = Preprocessing.dataloader.Program_Graph_Dataset('dataset/test')
    good_data_indices = [i for i, data in enumerate(dataset) if data[0][-1] == 2]
    filtered_dataset = Subset(dataset, good_data_indices)
    print(f"Total number of sketch data: {len(filtered_dataset)}")

    # Split the dataset into training and validation sets
    train_size = int(0.8 * len(filtered_dataset))
    val_size = len(filtered_dataset) - train_size
    train_dataset, val_dataset = random_split(filtered_dataset, [train_size, val_size])

    # Create DataLoaders for training and validation
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)

    # Training and validation loop
    best_val_loss = float('inf')
    epochs = 30

    for epoch in range(epochs):
        graph_encoder.train()
        graph_decoder.train()
        total_train_loss = 0.0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Training"):
            program, node_features, operations_order_matrix, gnn_strokeCloud_edges, face_to_stroke, stroke_cloud_coplanar, brep_to_stroke, edge_features, gnn_brep_edges, brep_stroke_connection, brep_coplanar = Preprocessing.dataloader.process_batch(batch)

            # Prev sketch
            sketch_op_index = len(program[0]) - 2
            prev_sketch_strokes = Encoders.helper.get_kth_operation(operations_order_matrix, sketch_op_index).to(device)

            # Current extrude
            target_op_index = len(program[0]) - 1
            extrude_strokes_raw = Encoders.helper.get_kth_operation(operations_order_matrix, target_op_index).to(device)
            extrude_lines = Models.sketch_model_helper.choose_extrude_strokes(prev_sketch_strokes, extrude_strokes_raw, node_features)

            # Create graph
            intersection_matrix = Encoders.helper.build_intersection_matrix(node_features).to(torch.float32).to(device)
            
            gnn_graph = Preprocessing.gnn_graph_stroke.SketchHeteroData(node_features, intersection_matrix)
            gnn_graph.set_brep_connection(edge_features)

            # Forward pass
            x_dict = graph_encoder(gnn_graph.x_dict, gnn_graph.edge_index_dict)
            output = graph_decoder(x_dict, gnn_graph.edge_index_dict, prev_sketch_strokes)
            
            # Models.sketch_model_helper.vis_gt_strokes(node_features, prev_sketch_strokes)
            # Models.sketch_model_helper.vis_gt_strokes(node_features, extrude_lines)

            loss = loss_function(output, extrude_lines)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_train_loss += loss.item()
        avg_train_loss = total_train_loss / len(train_loader)



        # Validation loop
        graph_encoder.eval()
        graph_decoder.eval()
        total_val_loss = 0.0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} - Validation"):
                program, node_features, operations_order_matrix, gnn_strokeCloud_edges, face_to_stroke, stroke_cloud_coplanar, brep_to_stroke, edge_features, gnn_brep_edges, brep_stroke_connection, brep_coplanar = Preprocessing.dataloader.process_batch(batch)

                # Prev sketch
                sketch_op_index = len(program[0]) - 2
                prev_sketch_strokes = Encoders.helper.get_kth_operation(operations_order_matrix, sketch_op_index).to(device)

                # Current extrude
                target_op_index = len(program[0]) - 1
                extrude_strokes_raw = Encoders.helper.get_kth_operation(operations_order_matrix, target_op_index).to(device)
                extrude_lines = Models.sketch_model_helper.choose_extrude_strokes(prev_sketch_strokes, extrude_strokes_raw, node_features)

                # Create graph
                intersection_matrix = Encoders.helper.build_intersection_matrix(node_features).to(torch.float32).to(device)
                
                gnn_graph = Preprocessing.gnn_graph_stroke.SketchHeteroData(node_features, intersection_matrix)
                gnn_graph.set_brep_connection(edge_features)

                # Forward pass
                x_dict = graph_encoder(gnn_graph.x_dict, gnn_graph.edge_index_dict)
                output = graph_decoder(x_dict, gnn_graph.edge_index_dict, prev_sketch_strokes)
                
                # Models.sketch_model_helper.vis_gt_strokes(node_features, prev_sketch_strokes)
                # Models.sketch_model_helper.vis_gt_strokes(node_features, extrude_lines)

                loss = loss_function(output, extrude_lines)
                # Models.sketch_model_helper.vis_gt_strokes(node_features, sketch_strokes)
                # Models.sketch_model_helper.vis_gt_strokes(node_features, output)

                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        if best_val_loss > total_val_loss:
            best_val_loss =  total_val_loss
            save_models()

        print(f"Epoch {epoch+1}/{epochs}: Avg Train Loss: {avg_train_loss:.4f}, Avg Val Loss: {avg_val_loss:.4f}")


def eval():
    load_models()
    dataset = Preprocessing.dataloader.Program_Graph_Dataset('dataset/eval')
    good_data_indices = [i for i, data in enumerate(dataset) if data[0][-1] == 2]
    filtered_dataset = Subset(dataset, good_data_indices)
    print(f"Total number of sketch data: {len(filtered_dataset)}")

    eval_loader = DataLoader(filtered_dataset, batch_size=1, shuffle=True)

    # Training and validation loop
    best_val_loss = float('inf')
    epochs = 30

    total_prediction = 0
    correct_prediction = 0

    
    for batch in tqdm(eval_loader, desc=f"Evaluating"):
        program, node_features, operations_order_matrix, gnn_strokeCloud_edges, face_to_stroke, stroke_cloud_coplanar, brep_to_stroke, edge_features, gnn_brep_edges, brep_stroke_connection, brep_coplanar = Preprocessing.dataloader.process_batch(batch)
        
        # Prev sketch
        sketch_op_index = len(program[0]) - 2
        prev_sketch_strokes = Encoders.helper.get_kth_operation(operations_order_matrix, sketch_op_index).to(device)

        # Current extrude
        target_op_index = len(program[0]) - 1
        extrude_strokes_raw = Encoders.helper.get_kth_operation(operations_order_matrix, target_op_index).to(device)
        extrude_lines = Models.sketch_model_helper.choose_extrude_strokes(prev_sketch_strokes, extrude_strokes_raw, node_features)

        # Create graph
        intersection_matrix = Encoders.helper.build_intersection_matrix(node_features).to(torch.float32).to(device)
        
        gnn_graph = Preprocessing.gnn_graph_stroke.SketchHeteroData(node_features, intersection_matrix)
        gnn_graph.set_brep_connection(edge_features)

        # Forward pass
        x_dict = graph_encoder(gnn_graph.x_dict, gnn_graph.edge_index_dict)
        output = graph_decoder(x_dict, gnn_graph.edge_index_dict, prev_sketch_strokes)
        
        # Models.sketch_model_helper.vis_gt_strokes(node_features, prev_sketch_strokes)
        # Models.sketch_model_helper.vis_gt_strokes(node_features, extrude_lines)
        # Models.sketch_model_helper.vis_gt_strokes(node_features, output)

        chosen_output = output > 0.5
        chosen_extrude_lines = extrude_lines > 0.5
        total_prediction += 1
        if torch.equal(chosen_output, chosen_extrude_lines):
            correct_prediction += 1
        
    accuracy = correct_prediction / total_prediction
    print(f"Model accuracy: {accuracy:.4f}")



#---------------------------------- Public Functions ----------------------------------#

eval()