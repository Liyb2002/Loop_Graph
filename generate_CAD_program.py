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
    
    # We only want to process complicated shapes
    if len(stroke_cloud_loops)< 60:
        continue
    
    # Init Brep
    brep_edges = torch.zeros(0)
    brep_loops = []


    # Strokes in the Graph
    stroke_in_graph = 0
    while stroke_in_graph < stroke_node_features.shape[0]:
    
    # -------------------- Prepare the graph informations -------------------- #
        # 1) Get stroke cloud loops
        read_strokes = stroke_node_features[:stroke_in_graph + 1]
        loops_fset = Preprocessing.proc_CAD.helper.face_aggregate_networkx(read_strokes)
        loops = [list(fset) for fset in loops_fset]

        # 2) Compute stroke / loop information 
        connected_stroke_nodes = Preprocessing.proc_CAD.helper.connected_strokes(read_strokes)
        loop_neighboring_all = Preprocessing.proc_CAD.helper.loop_neighboring_simple(loops)
        loop_neighboring_vertical = Preprocessing.proc_CAD.helper.loop_neighboring_complex(loops, read_strokes)
        loop_neighboring_horizontal = Preprocessing.proc_CAD.helper.coplanr_neighorbing_loop(loop_neighboring_all, loop_neighboring_vertical)
        loop_neighboring_contained = Preprocessing.proc_CAD.helper.loop_contained(loops, read_strokes)

        # 3) Stroke to Brep
        stroke_to_brep = Preprocessing.proc_CAD.helper.stroke_to_brep(loops, brep_loops, read_strokes, brep_edges)

        # 4) Build graph 
        gnn_graph = Preprocessing.gnn_graph.SketchLoopGraph(
            loops, 
            read_strokes, 
            connected_stroke_nodes,
            loop_neighboring_vertical, 
            loop_neighboring_horizontal, 
            loop_neighboring_contained,
            stroke_to_brep
        )


        stroke_in_graph += 1
        print("gnn_graph", gnn_graph['loop'].x.shape)
        print('------')
