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
save_dir = os.path.join(current_dir, 'checkpoints', 'program_prediction')
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



def compute_accuracy(output_stroke, output_loop, gt_token):
    
    # first value is the total (0 if not terminate)
    # second value is the correctness, 1 if terminate + correct prediction
    if gt_token != 0:
        return 0,0

    combined_logits = output_stroke + output_loop
    predicted_class = torch.argmax(combined_logits)
    
    if predicted_class == gt_token:
        return 1,1

    return 1,0


# ------------------------------------------------------------------------------# 


def train():
    # Load the dataset
    dataset = Preprocessing.dataloader.Program_Graph_Dataset('dataset/messy_order')
    print(f"Total number of shape data: {len(dataset)}")
    
    best_val_accuracy = 0
    epochs = 30
    
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
    # Split the dataset into training and validation sets (80-20 split)
    split_index = int(0.8 * len(graphs))
    train_graphs, val_graphs = graphs[:split_index], graphs[split_index:]
    train_existing_programs, val_existing_programs = existing_programs[:split_index], existing_programs[split_index:]
    train_gt_programs, val_gt_programs = gt_tokens[:split_index], gt_tokens[split_index:]



    # Training loop
    for epoch in range(epochs):
        train_loss = 0.0

        program_encoder.train()
        graph_encoder.train()
        graph_decoder_stroke.train()
        graph_decoder_loop.train()

        train_correct = 0
        train_total = 0


        total_iterations = len(train_existing_programs)
        for graph, program_existing, gt_token in tqdm(zip(train_graphs, train_existing_programs, train_gt_programs), 
                                              desc=f"Epoch {epoch+1}/{epochs} - Training", 
                                              dynamic_ncols=True, 
                                              total=total_iterations):

            optimizer.zero_grad()

            x_dict = graph_encoder(graph.x_dict, graph.edge_index_dict)
            output_loop = graph_decoder_loop(x_dict, program_existing)
            output_stroke = graph_decoder_loop(x_dict, program_existing)

            loss = criterion(output_loop.unsqueeze(0), gt_token) + criterion(output_stroke.unsqueeze(0), gt_token)

            tempt_total, tempt_correct = compute_accuracy(output_stroke, output_loop, gt_token)
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
            total_iterations_val = len(val_existing_programs)

            for graph, program_existing, gt_token in tqdm(zip(val_graphs, val_existing_programs, val_gt_programs), 
                                                desc=f"Epoch {epoch+1}/{epochs} - Validation", 
                                                dynamic_ncols=True, 
                                              total=total_iterations_val):
                
                x_dict = graph_encoder(graph.x_dict, graph.edge_index_dict)
                output_loop = graph_decoder_loop(x_dict, program_existing)
                output_stroke = graph_decoder_loop(x_dict, program_existing)

                loss = criterion(output_loop.unsqueeze(0), gt_token) + criterion(output_stroke.unsqueeze(0), gt_token)

                tempt_total, tempt_correct = compute_accuracy(output_stroke, output_loop, gt_token)
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