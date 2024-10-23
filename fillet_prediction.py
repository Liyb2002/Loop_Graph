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
graph_decoder = Encoders.gnn.gnn.Fillet_Decoder()

graph_encoder.to(device)
graph_decoder.to(device)

criterion = nn.BCELoss()
optimizer = optim.Adam(list(graph_encoder.parameters()) + list(graph_decoder.parameters()), lr=0.0004)
batch_size = 16
# ------------------------------------------------------------------------------# 

current_dir = os.getcwd()
save_dir = os.path.join(current_dir, 'checkpoints', 'fillet_prediction')
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
    dataset = Preprocessing.dataloader.Program_Graph_Dataset('dataset/test')
    print(f"Total number of shape data: {len(dataset)}")

    best_val_accuracy = 0
    epochs = 30

    graphs = []
    stroke_selection_masks = []

    # Preprocess and build the graphs
    for data in tqdm(dataset, desc=f"Building Graphs"):
        # Extract the necessary elements from the dataset
        program, program_whole, stroke_cloud_loops, stroke_node_features, strokes_perpendicular, output_brep_edges, stroke_operations_order_matrix, loop_neighboring_vertical, loop_neighboring_horizontal,loop_neighboring_contained, stroke_to_loop, stroke_to_edge = data

        if program[-1] != 'fillet'or len(program) > stroke_operations_order_matrix.shape[1]:
            continue
        
        kth_operation = Encoders.helper.get_kth_operation(stroke_operations_order_matrix, len(program)-1)
        raw_fillet_stroke_idx = (kth_operation == 1).nonzero(as_tuple=True)[0] 
        fillet_stroke_idx = Encoders.helper.choose_fillet_strokes(raw_fillet_stroke_idx, stroke_node_features)




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

        # Encoders.helper.vis_selected_loops(gnn_graph['stroke'].x.cpu().numpy(), gnn_graph['stroke', 'represents', 'loop'].edge_index, [torch.argmax(sketch_loop_selection_mask)])
        Encoders.helper.vis_selected_strokes(gnn_graph['stroke'].x.cpu().numpy(), fillet_stroke_idx)


    print(f"Total number of preprocessed graphs: {len(graphs)}")



#---------------------------------- Public Functions ----------------------------------#


train()