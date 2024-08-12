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
from sklearn.metrics import confusion_matrix


loop_embed_model = Models.loop_embeddings.LoopEmbeddingNetwork()
graph_encoder = Encoders.gnn.gnn.SemanticModule()
graph_decoder = Encoders.gnn.gnn.Program_prediction()
loop_embed_model.to(device)
graph_encoder.to(device)
graph_decoder.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(list(loop_embed_model.parameters()) + list(graph_encoder.parameters()) + list(graph_decoder.parameters()), lr=0.0004)

# ------------------------------------------------------------------------------# 

current_dir = os.getcwd()
save_dir = os.path.join(current_dir, 'checkpoints', 'program_prediction')
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
    
    # Split the dataset into training and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

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
            brep_loop_embeddings = loop_embed_model(edge_features, brep_to_stroke)
            print("gnn_brep_edges", gnn_brep_edges)
            # Build graph
            gnn_graph = Preprocessing.gnn_graph.SketchHeteroData(sketch_loop_embeddings, brep_loop_embeddings, gnn_strokeCloud_edges, gnn_brep_edges, brep_stroke_connection, stroke_cloud_coplanar, brep_coplanar)
            x_dict = graph_encoder(gnn_graph.x_dict, gnn_graph.edge_index_dict)

            # Prepare Program
            gt_next_token = program[0][-1]
            current_program = program[0][:-1]

            # Predict
            prediction = graph_decoder(x_dict, current_program).squeeze(0)

            # Loss
            loss = criterion(prediction, gt_next_token)
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

                # Loop embeddings
                sketch_loop_embeddings = loop_embed_model(node_features, face_to_stroke)
                brep_loop_embeddings = loop_embed_model(edge_features, brep_to_stroke)

                # Build graph
                gnn_graph = Preprocessing.gnn_graph.SketchHeteroData(sketch_loop_embeddings, brep_loop_embeddings, gnn_strokeCloud_edges, gnn_brep_edges, brep_stroke_connection, stroke_cloud_coplanar, brep_coplanar)
                x_dict = graph_encoder(gnn_graph.x_dict, gnn_graph.edge_index_dict)

                # Prepare Program
                gt_next_token = program[0][-1]
                current_program = program[0][:-1]

                # Predict
                prediction = graph_decoder(x_dict, current_program).squeeze(0)

                # Loss
                val_loss = criterion(prediction, gt_next_token)
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
    eval_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False)

    loop_embed_model.eval()
    graph_encoder.eval()
    graph_decoder.eval()

    total_eval_loss = 0.0
    total_predictions = 0
    correct_predictions = 0
    all_gt_tokens = []
    all_predicted_tokens = []

    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="Evaluation"):
            program, node_features, operations_order_matrix, gnn_strokeCloud_edges, face_to_stroke, stroke_cloud_coplanar, brep_to_stroke, edge_features, gnn_brep_edges, brep_stroke_connection, brep_coplanar = Preprocessing.dataloader.process_batch(batch)

            # Loop embeddings
            sketch_loop_embeddings = loop_embed_model(node_features, face_to_stroke)
            brep_loop_embeddings = loop_embed_model(edge_features, brep_to_stroke)

            # Build graph
            gnn_graph = Preprocessing.gnn_graph.SketchHeteroData(sketch_loop_embeddings, brep_loop_embeddings, gnn_strokeCloud_edges, gnn_brep_edges, brep_stroke_connection, stroke_cloud_coplanar, brep_coplanar)
            x_dict = graph_encoder(gnn_graph.x_dict, gnn_graph.edge_index_dict)

            # Prepare Program
            gt_next_token = program[0][-1]
            current_program = program[0][:-1]

            # Predict
            prediction = graph_decoder(x_dict, current_program).squeeze(0)
            predicted_token = torch.argmax(prediction)

            total_predictions += 1
            all_gt_tokens.append(gt_next_token.item())
            all_predicted_tokens.append(predicted_token.item())
            
            # Check if the prediction is correct
            if predicted_token == gt_next_token:
                correct_predictions += 1
            
    avg_eval_loss = total_eval_loss / total_predictions
    accuracy = correct_predictions / total_predictions

    print(f"Average evaluation loss: {avg_eval_loss:.4f}")
    print(f"Model accuracy: {accuracy:.4f}")

    # Compute and print confusion matrix
    cm = confusion_matrix(all_gt_tokens, all_predicted_tokens)
    print("Confusion Matrix:")
    print(cm)


#---------------------------------- Public Functions ----------------------------------#

train()