from torch.utils.data import Dataset, DataLoader
import os
import pickle
from tqdm import tqdm
import torch

import Preprocessing.SBGCN.brep_read
import Preprocessing.proc_CAD.helper
import Encoders.helper

import whole_process_evaluate




import Preprocessing.dataloader
import Preprocessing.gnn_graph

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
batch_size = 16

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



def compute_accuracy(valid_output, valid_batch_masks):
    batch_size = valid_output.shape[0] // 400
    correct = 0

    for i in range(batch_size):
        output_slice = valid_output[i * 400:(i + 1) * 400]
        mask_slice = valid_batch_masks[i * 400:(i + 1) * 400]

        condition_1 = (mask_slice == 1) & (output_slice > 0.5)
        condition_2 = (mask_slice == 0) & (output_slice < 0.5)


        if torch.all(condition_1 | condition_2):
            correct += 1

    return correct

# ------------------------------------------------------------------------------# 


def train():

    # Set up dataloader
    dataset = whole_process_evaluate.Evaluation_Dataset('program_output')

    total_correct = 0
    total = 0

    best_val_accuracy = 0
    epochs = 30

    graphs = []
    gt_on_right_track = []


    for data in tqdm(dataset, desc="Evaluating CAD Programs"):
        stroke_node_features, output_brep_edges, gt_brep_edges, on_right_track, is_finished, high_dist_indices, stroke_cloud_loops, strokes_perpendicular, loop_neighboring_vertical, loop_neighboring_horizontal, loop_neighboring_contained, stroke_to_loop, stroke_to_edge = data
        
        print("on_right_track?", on_right_track)

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



train()