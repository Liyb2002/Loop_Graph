import Preprocessing.dataloader
import Preprocessing.gnn_graph
import Encoders.gnn.gnn
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

import Models.loop_embeddings


loop_embed_model = Models.loop_embeddings.LoopEmbeddingNetwork()
graph_encoder = Encoders.gnn.gnn.SemanticModule()
graph_decoder = Encoders.gnn.gnn.Sketch_brep_prediction_timeEmbed()
loop_embed_model.to(device)
graph_encoder.to(device)
graph_decoder.to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(list(loop_embed_model.parameters()) + list(graph_encoder.parameters()) + list(graph_decoder.parameters()), lr=0.0004)

# ------------------------------------------------------------------------------# 

current_dir = os.getcwd()
save_dir = os.path.join(current_dir, 'checkpoints', 'sketch_prediction_timeEmbed')
os.makedirs(save_dir, exist_ok=True)

def load_models():
    loop_embed_model.load_state_dict(torch.load(os.path.join(save_dir, 'loop_embed_model.pth')))
    graph_encoder.load_state_dict(torch.load(os.path.join(save_dir, 'graph_encoder.pth')))
    graph_decoder.load_state_dict(torch.load(os.path.join(save_dir, 'graph_decoder.pth')))


def save_models():
    torch.save(loop_embed_model.state_dict(), os.path.join(save_dir, 'loop_embed_model.pth'))
    torch.save(graph_encoder.state_dict(), os.path.join(save_dir, 'graph_encoder.pth'))
    torch.save(graph_decoder.state_dict(), os.path.join(save_dir, 'graph_decoder.pth'))


# ------------------------------------------------------------------------------# 



def train():
    # Load the dataset
    dataset = Preprocessing.dataloader.Program_Graph_Dataset('dataset/test')
    good_data_indices = [i for i, data in enumerate(dataset) if data[0][-1] == 1]
    filtered_dataset = Subset(dataset, good_data_indices)
    print(f"Total number of sketch data: {len(filtered_dataset)}")

    # Split the dataset into training and validation sets
    train_size = int(0.8 * len(filtered_dataset))
    val_size = len(filtered_dataset) - train_size
    train_dataset, val_dataset = random_split(filtered_dataset, [train_size, val_size])

    # Create DataLoaders for training and validation
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)

    best_val_loss = float('inf')
    epochs = 30

    for epoch in range(epochs):
        total_train_loss = 0.0
        loop_embed_model.train()
        graph_encoder.train()
        graph_decoder.train()

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Training"):
            program, node_features, operations_order_matrix, gnn_strokeCloud_edges, face_to_stroke, stroke_cloud_coplanar, brep_to_stroke, edge_features, gnn_brep_edges, brep_stroke_connection, brep_coplanar = Preprocessing.dataloader.process_batch(batch)
            optimizer.zero_grad()

            # Loop embeddings
            sketch_loop_embeddings = loop_embed_model(node_features, face_to_stroke)

            # Build graph
            gnn_graph = Preprocessing.gnn_graph.SketchHeteroData(sketch_loop_embeddings, gnn_strokeCloud_edges, stroke_cloud_coplanar, brep_stroke_connection)
            x_dict = graph_encoder(gnn_graph.x_dict, gnn_graph.edge_index_dict)
            # Predict
            prediction = graph_decoder(x_dict, face_to_stroke).squeeze(0)


            # Prepare Ground_truth
            target_op_index = len(program[0]) - 1
            op_to_index_matrix = operations_order_matrix
            kth_operation = Encoders.helper.get_kth_operation(op_to_index_matrix, target_op_index).to(device)
            face_choice = Encoders.helper.stroke_to_face(kth_operation, face_to_stroke).float()

            # Loss
            loss = criterion(prediction, face_choice)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
        avg_train_loss = total_train_loss / len(train_loader)

        print(f"Epoch {epoch+1}/{epochs} - avg_train_loss: {avg_train_loss:.4f}")


        # Validation
        total_val_loss = 0.0
        loop_embed_model.eval()
        graph_encoder.eval()
        graph_decoder.eval()

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} - Validation"):
                program, node_features, operations_order_matrix, gnn_strokeCloud_edges, face_to_stroke, stroke_cloud_coplanar, brep_to_stroke, edge_features, gnn_brep_edges, brep_stroke_connection, brep_coplanar = Preprocessing.dataloader.process_batch(batch)
                optimizer.zero_grad()

                # Loop embeddings
                sketch_loop_embeddings = loop_embed_model(node_features, face_to_stroke)

                # Build graph
                gnn_graph = Preprocessing.gnn_graph.SketchHeteroData(sketch_loop_embeddings, gnn_strokeCloud_edges, stroke_cloud_coplanar, brep_stroke_connection)
                x_dict = graph_encoder(gnn_graph.x_dict, gnn_graph.edge_index_dict)
                # Predict
                prediction = graph_decoder(x_dict, face_to_stroke).squeeze(0)


                # Prepare Ground_truth
                target_op_index = len(program[0]) - 1
                op_to_index_matrix = operations_order_matrix
                kth_operation = Encoders.helper.get_kth_operation(op_to_index_matrix, target_op_index).to(device)
                face_choice = Encoders.helper.stroke_to_face(kth_operation, face_to_stroke).float()

                # Loss
                val_loss = criterion(prediction, face_choice)
                total_val_loss += val_loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        print(f"Epoch {epoch+1}/{epochs} - avg_val_loss: {avg_val_loss:.4f}")

        # Save models if validation loss improves
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_models()
            print(f"Model saved at epoch {epoch+1} with val_loss: {avg_val_loss:.4f}")


def eval():
    load_models()

    # Load the evaluation dataset
    eval_dataset = Preprocessing.dataloader.Program_Graph_Dataset('dataset/eval')
    good_data_indices = [i for i, data in enumerate(eval_dataset) if data[0][-1] == 1]
    filtered_eval_dataset = Subset(eval_dataset, good_data_indices)
    print(f"Total number of evaluation sketch data: {len(filtered_eval_dataset)}")

    # Create DataLoader for evaluation
    eval_loader = DataLoader(filtered_eval_dataset, batch_size=1, shuffle=False)

    loop_embed_model.eval()
    graph_encoder.eval()
    graph_decoder.eval()

    total_eval_loss = 0.0
    total_predictions = 0
    correct_predictions = 0
    valid_predictions = 0

    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="Evaluation"):
            program, node_features, operations_order_matrix, gnn_strokeCloud_edges, face_to_stroke, stroke_cloud_coplanar, brep_to_stroke, edge_features, gnn_brep_edges, brep_stroke_connection, brep_coplanar = Preprocessing.dataloader.process_batch(batch)
            optimizer.zero_grad()

            # Loop embeddings
            sketch_loop_embeddings = loop_embed_model(node_features, face_to_stroke)

            # Build graph
            gnn_graph = Preprocessing.gnn_graph.SketchHeteroData(sketch_loop_embeddings, gnn_strokeCloud_edges, stroke_cloud_coplanar, brep_stroke_connection)
            x_dict = graph_encoder(gnn_graph.x_dict, gnn_graph.edge_index_dict)
            # Predict
            prediction = graph_decoder(x_dict, face_to_stroke).squeeze(0)


            # Prepare Ground_truth
            target_op_index = len(program[0]) - 1
            op_to_index_matrix = operations_order_matrix
            kth_operation = Encoders.helper.get_kth_operation(op_to_index_matrix, target_op_index).to(device)
            face_choice = Encoders.helper.stroke_to_face(kth_operation, face_to_stroke).float()

            # Calculate loss
            eval_loss = criterion(prediction, face_choice)
            total_eval_loss += eval_loss.item()

            # Find the predicted face
            predicted_face_index = torch.argmax(prediction).item()
            ground_truth_face_index = torch.argmax(face_choice).item()

            # Vis
            # Encoders.helper.vis_brep_and_nextSketch(face_choice, face_to_stroke, node_features, edge_features)
            # Encoders.helper.vis_brep_and_nextSketch(prediction, face_to_stroke, node_features, edge_features)

            # Encoders.helper.vis_stroke_cloud(node_features)
            # Encoders.helper.vis_stroke_cloud(edge_features)
            # Encoders.helper.vis_gt(face_choice, face_to_stroke, node_features)
            # Encoders.helper.vis_gt(prediction, face_to_stroke, node_features)

            if len(program[0]) == 7:
                total_predictions += 1
                # Check if the prediction is correct
                if predicted_face_index == ground_truth_face_index:
                    correct_predictions += 1
                else:
                    Encoders.helper.vis_brep_and_nextSketch(face_choice, face_to_stroke, node_features, edge_features)
                    Encoders.helper.vis_brep_and_nextSketch(prediction, face_to_stroke, node_features, edge_features)
                    Encoders.helper.vis_brep_and_nextSketch(face_choice, face_to_stroke, node_features, edge_features)
                    Encoders.helper.vis_brep_and_nextSketch(prediction, face_to_stroke, node_features, edge_features)



    avg_eval_loss = total_eval_loss / total_predictions
    accuracy = correct_predictions / total_predictions

    print(f"Average evaluation loss: {avg_eval_loss:.4f}")
    print(f"Model accuracy: {accuracy:.4f}")


#---------------------------------- Public Functions ----------------------------------#

eval()