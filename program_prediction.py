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
    combined_logits = output_stroke + output_loop
    predicted_class = torch.argmax(combined_logits)

    if predicted_class == gt_token:
        return 1

    return 0


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

            train_correct += compute_accuracy(output_stroke, output_loop, gt_token)
            train_total += 1


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

                val_correct += compute_accuracy(output_stroke, output_loop, gt_token)
                val_total += 1

        
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


    eval_graphs = []
    eval_loop_selection_masks = []
    eval_all_loop_selection_masks = []

    # Preprocess and build the graphs
    for data in tqdm(dataset, desc=f"Building Graphs"):
        # Extract the necessary elements from the dataset
        program, program_whole, stroke_cloud_loops, stroke_node_features, strokes_perpendicular, output_brep_edges, stroke_operations_order_matrix, loop_neighboring_vertical, loop_neighboring_horizontal,loop_neighboring_contained, stroke_to_loop, stroke_to_edge = data

        if program[-1] != 'sketch':
            continue

        kth_operation = Encoders.helper.get_kth_operation(stroke_operations_order_matrix, len(program)-1)
        all_sketch_strokes = Encoders.helper.get_all_operation_strokes(stroke_operations_order_matrix, program_whole, 'sketch')

        # Gets the strokes for the current sketch Operation
        chosen_strokes = (kth_operation == 1).nonzero(as_tuple=True)[0]  # Indices of chosen stroke
        loop_chosen_mask = []
        for loop in stroke_cloud_loops:
            if all(stroke in chosen_strokes for stroke in loop):
                loop_chosen_mask.append(1)  # Loop is chosen
            else:
                loop_chosen_mask.append(0)  # Loop is not chosen
        
        loop_selection_mask = torch.tensor(loop_chosen_mask, dtype=torch.float).reshape(-1, 1)
        if not (loop_selection_mask == 1).any():
            continue

        
        # Gets the strokes for all sketch Operation
        all_chosen_strokes = (all_sketch_strokes == 1).nonzero(as_tuple=True)[0]  # Indices of chosen stroke
        all_loop_chosen_mask = []
        for loop in stroke_cloud_loops:
            if all(stroke in all_chosen_strokes for stroke in loop):
                all_loop_chosen_mask.append(1)  # Loop is chosen
            else:
                all_loop_chosen_mask.append(0)  # Loop is not chosen
        all_loop_selection_mask = torch.tensor(all_loop_chosen_mask, dtype=torch.float).reshape(-1, 1)

        
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
        all_selected_loops_idx = [idx for idx, value in enumerate(all_loop_chosen_mask) if value != 0]

        # Encoders.helper.vis_selected_loops(gnn_graph['stroke'].x.cpu().numpy(), gnn_graph['stroke', 'represents', 'loop'].edge_index, all_selected_loops_idx )

        # Prepare the pair
        gnn_graph.to_device_withPadding(device)
        loop_selection_mask = loop_selection_mask.to(device)

        eval_graphs.append(gnn_graph)
        eval_loop_selection_masks.append(loop_selection_mask)
        eval_all_loop_selection_masks.append(all_loop_selection_mask)


    print(f"Total number of preprocessed graphs: {len(eval_graphs)}")


    # Convert train and validation graphs to HeteroData
    hetero_eval_graphs = [Preprocessing.gnn_graph.convert_to_hetero_data(graph) for graph in eval_graphs]
    padded_eval_masks = [Preprocessing.dataloader.pad_masks(mask) for mask in eval_loop_selection_masks]
    padded_eval_all_masks = [Preprocessing.dataloader.pad_masks(mask) for mask in eval_all_loop_selection_masks]

    # Create DataLoaders for training and validation graphs/masks
    graph_eval_loader = DataLoader(hetero_eval_graphs, batch_size=16, shuffle=False)
    mask_eval_loader = DataLoader(padded_eval_masks, batch_size=16, shuffle=False)
    mask_eval_all_loader = DataLoader(padded_eval_all_masks, batch_size=16, shuffle=False)



    # Eval
    graph_encoder.eval()
    graph_decoder.eval()

    eval_loss = 0.0
    total_category_count = [0, 0, 0, 0]
    total_correct_count = [0, 0, 0, 0] 

    with torch.no_grad():
        total_iterations_eval = min(len(graph_eval_loader), len(mask_eval_all_loader))

        for hetero_batch, batch_masks, in tqdm(zip(graph_eval_loader, mask_eval_all_loader), 
                                                desc="Evaluation", 
                                                dynamic_ncols=True, 
                                                total=total_iterations_eval):
                        

            x_dict = graph_encoder(hetero_batch.x_dict, hetero_batch.edge_index_dict)
            output = graph_decoder(x_dict)

            batch_masks = batch_masks.to(output.device).view(-1, 1)
            valid_mask = (batch_masks != -1).float()
            valid_output = output * valid_mask
            valid_batch_masks = batch_masks * valid_mask


            category_count, correct_count = compute_accuracy_with_lvl(valid_output, valid_batch_masks, hetero_batch)           

            for i in range(4):
                total_category_count[i] += category_count[i]
                total_correct_count[i] += correct_count[i]

            # Compute loss
            loss = criterion(valid_output, valid_batch_masks)
            eval_loss += loss.item()


    print("Category-wise Accuracy:")
    total_correct = 0
    total_samples = 0

    for i in range(4):
        if total_category_count[i] > 0:
            accuracy = total_correct_count[i] / total_category_count[i] * 100
            print(f"Category {i+1}: {accuracy:.2f}% (Correct: {total_correct_count[i]}/{total_category_count[i]})")
        else:
            print(f"Category {i+1}: No samples")

    # Calculate and print average evaluation loss
    average_eval_loss = eval_loss / total_iterations_eval
    print(f"Average Evaluation Loss: {average_eval_loss:.4f}")

    # Calculate and print overall average accuracy
    overall_accuracy = total_correct / total_samples * 100
    print(f"Overall Average Accuracy: {overall_accuracy:.2f}%")




#---------------------------------- Public Functions ----------------------------------#


train()