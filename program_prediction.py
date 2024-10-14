import Preprocessing.dataloader
import Preprocessing.gnn_graph
import Preprocessing.gnn_graph_stroke

import Encoders.gnn.gnn
import Encoders.helper

from torch_geometric.loader import DataLoader as GraphDataLoader
from torch.utils.data import DataLoader, TensorDataset

from tqdm import tqdm
from Preprocessing.config import device
import torch
import torch.nn as nn
import torch.optim as optim
import os
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

program_encoder = Encoders.gnn.gnn.ProgramEncoder()
graph_encoder = Encoders.gnn.gnn.SemanticModule()
graph_decoder_stroke = Encoders.gnn.gnn.Program_Decoder('stroke')
graph_decoder_loop = Encoders.gnn.gnn.Program_Decoder('loop')

program_encoder.to(device)
graph_encoder.to(device)
graph_decoder_stroke.to(device)
graph_decoder_loop.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(list(program_encoder.parameters()) + list(graph_encoder.parameters()) + list(graph_decoder_stroke.parameters()) + list(graph_decoder_loop.parameters()), lr=0.0004)

# ------------------------------------------------------------------------------# 

current_dir = os.getcwd()
save_dir = os.path.join(current_dir, 'checkpoints', 'program_prediction_2')
os.makedirs(save_dir, exist_ok=True)

def load_models():
    program_encoder.load_state_dict(torch.load(os.path.join(save_dir, 'program_encoder.pth')))
    graph_encoder.load_state_dict(torch.load(os.path.join(save_dir, 'graph_encoder.pth')))
    graph_decoder_stroke.load_state_dict(torch.load(os.path.join(save_dir, 'graph_decoder.pth')))
    graph_decoder_loop.load_state_dict(torch.load(os.path.join(save_dir, 'graph_decoder.pth')))


def save_models():
    torch.save(program_encoder.state_dict(), os.path.join(save_dir, 'program_encoder.pth'))
    torch.save(graph_encoder.state_dict(), os.path.join(save_dir, 'graph_encoder.pth'))
    torch.save(graph_decoder_stroke.state_dict(), os.path.join(save_dir, 'graph_decoder.pth'))
    torch.save(graph_decoder_loop.state_dict(), os.path.join(save_dir, 'graph_decoder.pth'))


# ------------------------------------------------------------------------------# 



def compute_accuracy(output_stroke, output_loop, program_gt_batch):
    # Get the predicted classes from the output
    predicted_classes = torch.argmax(output_stroke + output_loop, dim=1)
    
    # Flatten the ground truth labels for comparison
    program_gt_batch = program_gt_batch.view(-1)
    
    # Create a mask to ignore cases where the ground truth is 2
    valid_mask = program_gt_batch != 2
    
    # Apply the mask to both the predicted and ground truth tensors
    filtered_predicted_classes = predicted_classes[valid_mask]
    filtered_program_gt_batch = program_gt_batch[valid_mask]
    
    # Calculate the number of correct predictions
    correct_predictions = (filtered_predicted_classes == filtered_program_gt_batch).sum().item()
    
    # Calculate the total number of valid cases
    total = filtered_program_gt_batch.shape[0]
    
    # Output the total and the correct number of predictions
    return total, correct_predictions


# ------------------------------------------------------------------------------# 


def train():
    # Load the dataset
    dataset = Preprocessing.dataloader.Program_Graph_Dataset('dataset/messy_order')
    print(f"Total number of shape data: {len(dataset)}")
    
    best_val_accuracy = 0
    epochs = 30
    
    graphs = []
    existing_programs = []
    gt_programs = []

    # Preprocess and build the graphs
    for data in tqdm(dataset, desc=f"Building Graphs"):
        # Extract the necessary elements from the dataset
        program, program_whole, stroke_cloud_loops, stroke_node_features, strokes_perpendicular, output_brep_edges, stroke_operations_order_matrix, loop_neighboring_vertical, loop_neighboring_horizontal,loop_neighboring_contained, stroke_to_loop, stroke_to_edge = data
        
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

        graphs.append(gnn_graph)
        
        existing_programs.append(Encoders.helper.program_mapping(program[:-1]))
        gt_programs.append(Encoders.helper.program_gt_mapping([program[-1]]))



    print(f"Total number of preprocessed graphs: {len(graphs)}")
    # Split the dataset into training and validation sets (80-20 split)
    split_index = int(0.8 * len(graphs))
    train_graphs, val_graphs = graphs[:split_index], graphs[split_index:]
    train_existing_programs, val_existing_programs = existing_programs[:split_index], existing_programs[split_index:]
    train_gt_programs, val_gt_programs = gt_programs[:split_index], gt_programs[split_index:]


    # Convert train and validation graphs to HeteroData
    hetero_train_graphs = [Preprocessing.gnn_graph.convert_to_hetero_data(graph) for graph in train_graphs]
    hetero_val_graphs = [Preprocessing.gnn_graph.convert_to_hetero_data(graph) for graph in val_graphs]
    graph_train_loader = GraphDataLoader(hetero_train_graphs, batch_size=16, shuffle=False)
    graph_val_loader = GraphDataLoader(hetero_val_graphs, batch_size=16, shuffle=False)


    train_existing_programs_tensor = torch.tensor(train_existing_programs, dtype=torch.float32)
    val_existing_programs_tensor = torch.tensor(val_existing_programs, dtype=torch.float32)
    train_existing_dataset = TensorDataset(train_existing_programs_tensor)
    val__existing_dataset = TensorDataset(val_existing_programs_tensor)
    program_train_existing_loader = DataLoader(train_existing_dataset, batch_size=16, shuffle=False)
    program_val_existing_loader = DataLoader(val__existing_dataset, batch_size=16, shuffle=False)


    train_gt_programs_tensor = torch.tensor(train_gt_programs, dtype=torch.float32)
    val_gt_programs_tensor = torch.tensor(val_gt_programs, dtype=torch.float32)
    train_gt_dataset = TensorDataset(train_gt_programs_tensor)
    val__gt_dataset = TensorDataset(val_gt_programs_tensor)
    program_train_gt_loader = DataLoader(train_gt_dataset, batch_size=16, shuffle=False)
    program_val_gt_loader = DataLoader(val__gt_dataset, batch_size=16, shuffle=False)


    # Training loop
    for epoch in range(epochs):
        train_loss = 0.0

        program_encoder.train()
        graph_encoder.train()
        graph_decoder_stroke.train()
        graph_decoder_loop.train()

        train_correct = 0
        train_total = 0


        total_iterations = min(len(graph_train_loader), len(program_train_existing_loader))
        for hetero_batch, (program_existing_batch,),  (program_gt_batch,) in tqdm(zip(graph_train_loader, program_train_existing_loader, program_train_gt_loader), 
                                              desc=f"Epoch {epoch+1}/{epochs} - Training", 
                                              dynamic_ncols=True, 
                                              total=total_iterations):

            optimizer.zero_grad()

            x_dict = graph_encoder(hetero_batch.x_dict, hetero_batch.edge_index_dict)
            output_loop = graph_decoder_loop(x_dict, program_existing_batch.long())
            output_stroke = graph_decoder_loop(x_dict, program_existing_batch.long())

            loss = criterion(output_loop, program_gt_batch.squeeze(1).long()) + criterion(output_stroke, program_gt_batch.squeeze(1).long())

            tempt_total, tempt_correct = compute_accuracy(output_stroke, output_loop, program_gt_batch)
            train_correct += tempt_correct
            train_total += tempt_total


            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_accuracy = train_correct / train_total if train_total > 0 else 0
        train_loss /= len(train_graphs)
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        program_encoder.eval()
        graph_encoder.eval()
        graph_decoder_stroke.eval()
        graph_decoder_loop.eval()
        with torch.no_grad():
            total_iterations_val = min(len(graph_val_loader), len(program_val_existing_loader))

            for hetero_batch, (program_existing_batch,),  (program_gt_batch,) in tqdm(zip(graph_val_loader, program_val_existing_loader, program_val_gt_loader), 
                                                desc=f"Epoch {epoch+1}/{epochs} - Validation", 
                                                dynamic_ncols=True, 
                                              total=total_iterations_val):
                
                x_dict = graph_encoder(hetero_batch.x_dict, hetero_batch.edge_index_dict)
                output_loop = graph_decoder_loop(x_dict, program_existing_batch.long())
                output_stroke = graph_decoder_loop(x_dict, program_existing_batch.long())

                loss = criterion(output_loop, program_gt_batch.squeeze(1).long()) + criterion(output_stroke, program_gt_batch.squeeze(1).long())

                tempt_total, tempt_correct = compute_accuracy(output_stroke, output_loop, program_gt_batch)
                val_correct += tempt_correct
                val_total += tempt_total

        
        val_loss /= len(val_graphs)
        

        val_accuracy = val_correct / val_total if val_total > 0 else 0
        print(f"Epoch {epoch+1}/{epochs} - Training Loss: {train_loss:.7f} - Validation Loss: {val_loss:.7f} - Train Accuracy: {train_accuracy:.5f} - Validation Accuracy: {val_accuracy:.5f}")

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            save_models()
            print(f"Models saved at epoch {epoch+1} with validation accuracy: {val_accuracy:.5f}")



def eval():
    load_models()
    # Load the dataset
    dataset = Preprocessing.dataloader.Program_Graph_Dataset('dataset/messy_order_eval')
    print(f"Total number of shape data: {len(dataset)}")


    graphs = []
    existing_programs = []
    gt_tokens = []

    # Preprocess and build the graphs
    for data in tqdm(dataset, desc=f"Building Graphs"):
        # Extract the necessary elements from the dataset
        program, program_whole, stroke_cloud_loops, stroke_node_features, strokes_perpendicular, output_brep_edges, stroke_operations_order_matrix, loop_neighboring_vertical, loop_neighboring_horizontal,loop_neighboring_contained, stroke_to_loop, stroke_to_edge = data


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

        gnn_graph.to_device(device)
        graphs.append(gnn_graph)
        
        existing_program = torch.tensor(Encoders.helper.program_mapping(program[:-1]), dtype=torch.long).to(device)
        gt_token = torch.tensor(Encoders.helper.program_gt_mapping([program[-1]]))
        existing_programs.append(existing_program)
        gt_tokens.append(gt_token)


    print(f"Total number of preprocessed graphs: {len(graphs)}")



    # Eval
    program_encoder.eval()
    graph_encoder.eval()
    graph_decoder_stroke.eval()
    graph_decoder_loop.eval()

    eval_correct = 0
    eval_total = 0

    with torch.no_grad():
        total_iterations_eval = len(gt_tokens)

        for graph, program_existing, gt_token in tqdm(zip(graphs, existing_programs, gt_tokens), 
                                                desc="Evaluation", 
                                                dynamic_ncols=True, 
                                                total=total_iterations_eval):
                        

            if gt_token != 0:
                continue

            x_dict = graph_encoder(graph.x_dict, graph.edge_index_dict)
            output_loop = graph_decoder_loop(x_dict, program_existing)
            output_stroke = graph_decoder_loop(x_dict, program_existing)

            # Compute Accuracy
            print("-----")
            print('program_existing', program_existing)
            tempt_total, tempt_correct = compute_accuracy(output_stroke, output_loop, gt_token)
            eval_correct += tempt_correct
            eval_total += tempt_total



    # Calculate and print overall average accuracy
    overall_accuracy = eval_correct / eval_total
    print(f"Overall Average Accuracy: {overall_accuracy:.4f}")




#---------------------------------- Public Functions ----------------------------------#


train()