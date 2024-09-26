import Preprocessing.dataloader
import Preprocessing.generate_dataset
import Preprocessing.gnn_graph
import Preprocessing.gnn_graph_stroke

import Preprocessing.proc_CAD.generate_program
import Preprocessing.proc_CAD.Program_to_STL
import Preprocessing.proc_CAD.brep_read
import Preprocessing.proc_CAD.helper

import Models.loop_embeddings

import Encoders.gnn.gnn
import Encoders.gnn_stroke.gnn
import Encoders.helper


from torch.utils.data import DataLoader, random_split, Subset
from tqdm import tqdm
from Preprocessing.config import device
import torch
import torch.nn as nn
import torch.optim as optim
import os
import shutil

import random

# --------------------- Dataset --------------------- #
dataset = Preprocessing.dataloader.Program_Graph_Dataset('dataset/simple')


# --------------------- Directory --------------------- #
current_dir = os.getcwd()
output_dir = os.path.join(current_dir, 'program_output')
output_relative_dir = ('program_output/canvas')



# --------------------- Skecth Network --------------------- #
pass

# --------------------- Extrude Network --------------------- #
pass



# --------------------- Cascade Brep Features --------------------- #
pass


# --------------------- Main Code --------------------- #
for data in tqdm(dataset, desc=f"Generating CAD Progams"):
    stroke_cloud_loops, stroke_node_features, connected_stroke_nodes, loop_neighboring_vertical, loop_neighboring_horizontal, loop_neighboring_contained, loop_neighboring_coplanar, stroke_to_brep, stroke_operations_order_matrix, final_brep_edges = data

    print("stroke_cloud_loops.shape[0]", len(stroke_cloud_loops))
    
    if len(stroke_cloud_loops)< 60:
        continue