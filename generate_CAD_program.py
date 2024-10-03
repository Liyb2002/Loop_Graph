import Preprocessing.dataloader
import Preprocessing.generate_dataset
import Preprocessing.gnn_graph
import Preprocessing.gnn_graph_stroke

import Preprocessing.proc_CAD.generate_program
import Preprocessing.proc_CAD.Program_to_STL
import Preprocessing.proc_CAD.brep_read
import Preprocessing.proc_CAD.helper

import whole_process_helper.helper

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
import numpy as np
import random

# --------------------- Dataset --------------------- #
dataset = Preprocessing.dataloader.Program_Graph_Dataset('dataset/messy_order')


# --------------------- Directory --------------------- #
current_dir = os.getcwd()
output_dir = os.path.join(current_dir, 'program_output')
output_relative_dir = ('program_output/canvas')



# --------------------- Skecth Network --------------------- #
sketch_graph_encoder = Encoders.gnn.gnn.SemanticModule()
sketch_graph_decoder = Encoders.gnn.gnn.Sketch_Decoder()
sketch_graph_encoder.eval()
sketch_graph_decoder.eval()
sketch_dir = os.path.join(current_dir, 'checkpoints', 'sketch_prediction')
sketch_graph_encoder.load_state_dict(torch.load(os.path.join(sketch_dir, 'graph_encoder.pth')))
sketch_graph_decoder.load_state_dict(torch.load(os.path.join(sketch_dir, 'graph_decoder.pth')))

def predict_sketch(gnn_graph):
    x_dict = sketch_graph_encoder(gnn_graph.x_dict, gnn_graph.edge_index_dict)
    sketch_selection_mask = sketch_graph_decoder(x_dict)
    # Encoders.helper.vis_whole_graph(gnn_graph, torch.argmax(sketch_selection_mask))

    return sketch_selection_mask

def do_sketch(gnn_graph):
    sketch_selection_mask = predict_sketch(gnn_graph)
    sketch_points = whole_process_helper.helper.extract_unique_points(sketch_selection_mask, gnn_graph)
    normal = [1, 0, 0]
    sketch_selection_mask = whole_process_helper.helper.clean_mask(sketch_selection_mask)

    return sketch_selection_mask, sketch_points, normal

# --------------------- Extrude Network --------------------- #
extrude_graph_encoder = Encoders.gnn.gnn.SemanticModule()
extrude_graph_decoder = Encoders.gnn.gnn.Extrude_Decoder()
extrude_dir = os.path.join(current_dir, 'checkpoints', 'extrude_prediction')
extrude_graph_encoder.eval()
extrude_graph_decoder.eval()
extrude_graph_encoder.load_state_dict(torch.load(os.path.join(extrude_dir, 'graph_encoder.pth')))
extrude_graph_decoder.load_state_dict(torch.load(os.path.join(extrude_dir, 'graph_decoder.pth')))

def predict_extrude(gnn_graph, sketch_selection_mask):
    gnn_graph.set_select_sketch(sketch_selection_mask)

    x_dict = extrude_graph_encoder(gnn_graph.x_dict, gnn_graph.edge_index_dict)
    extrude_selection_mask = extrude_graph_decoder(x_dict)
    # Encoders.helper.vis_stroke_graph(gnn_graph, extrude_selection_mask.detach())
    return extrude_selection_mask

# This extrude_amount, extrude_direction is not total correct. Work on it later
def do_extrude(gnn_graph, sketch_selection_mask, sketch_points, brep_edges):
    extrude_selection_mask = predict_extrude(gnn_graph, sketch_selection_mask)
    extrude_amount, extrude_direction = whole_process_helper.helper.get_extrude_amount(gnn_graph, extrude_selection_mask, sketch_points, brep_edges)
    return extrude_amount, extrude_direction


# --------------------- Cascade Brep Features --------------------- #
def cascade_brep(brep_files):
    final_brep_edges_list = []

    for file_name in brep_files:
        brep_directory = os.path.join(output_dir, 'canvas')
        brep_file_path = os.path.join(brep_directory, file_name)

        edge_features_list, _= Preprocessing.SBGCN.brep_read.create_graph_from_step_file(brep_file_path)
        if len(final_brep_edges_list) == 0:
            final_brep_edges_list = edge_features_list
            new_features = edge_features_list
        else:
            # We already have brep
            new_features= Preprocessing.generate_dataset.find_new_features(final_brep_edges_list, edge_features_list) 
            final_brep_edges_list += new_features

        Encoders.helper.vis_brep(np.array(new_features))
        Encoders.helper.vis_brep(np.array(final_brep_edges_list))

    final_brep_edges = np.array(final_brep_edges_list)
    final_brep_edges = np.round(final_brep_edges, 4)
    brep_loops = Preprocessing.proc_CAD.helper.face_aggregate_networkx(final_brep_edges)

    return final_brep_edges, brep_loops



# --------------------- Main Code --------------------- #
for data in tqdm(dataset, desc=f"Generating CAD Progams"):
    stroke_cloud_loops, stroke_node_features, _, _, _, _, _, _, _ ,_ = data
    
    print("NEW SHAPE -----------------!")
    # We only want to process complicated shapes
    if len(stroke_cloud_loops)< 30:
        continue
    
    # Init Brep
    brep_edges = torch.zeros(0)
    brep_loops = []
    file_path = os.path.join(output_dir, 'Program.json')


    # Program State Init
    cur__brep_class = Preprocessing.proc_CAD.generate_program.Brep()
    cur_program = torch.tensor([], dtype=torch.int64)


    # Strokes / Loops in the Graph
    stroke_in_graph = 0
    existing_loops = []

    while stroke_in_graph < stroke_node_features.shape[0]:
        print("stroke_in_graph", stroke_in_graph, "out of", stroke_node_features.shape[0])

    # -------------------- Prepare the graph informations -------------------- #
        # 1) Get stroke cloud loops
        read_strokes = stroke_node_features[:stroke_in_graph + 1]
        loops_fset = whole_process_helper.helper.face_aggregate_addStroke(read_strokes)
        existing_loops += [list(fset) for fset in loops_fset]

        # Preprocessing.proc_CAD.helper.vis_multiple_loops([list(fset) for fset in loops_fset], read_strokes)

        # 2) Compute stroke / loop information 
        connected_stroke_nodes = Preprocessing.proc_CAD.helper.connected_strokes(read_strokes)
        strokes_perpendicular, strokes_non_perpendicular =  Preprocessing.proc_CAD.helper.stroke_relations(read_strokes, connected_stroke_nodes)


        loop_neighboring_all = Preprocessing.proc_CAD.helper.loop_neighboring_simple(existing_loops)
        loop_neighboring_vertical = Preprocessing.proc_CAD.helper.loop_neighboring_complex(existing_loops, read_strokes)
        loop_neighboring_horizontal = Preprocessing.proc_CAD.helper.coplanr_neighorbing_loop(loop_neighboring_all, loop_neighboring_vertical)
        loop_neighboring_contained = Preprocessing.proc_CAD.helper.loop_contained(existing_loops, read_strokes)
        
        # 3) Stroke to Brep
        stroke_to_loop = Preprocessing.proc_CAD.helper.stroke_to_brep(existing_loops, brep_loops, read_strokes, brep_edges)
        stroke_to_edge = Preprocessing.proc_CAD.helper.stroke_to_edge(read_strokes, brep_edges)

        # 4) Build graph & check validity of the graph
        gnn_graph = Preprocessing.gnn_graph.SketchLoopGraph(
            existing_loops, 
            read_strokes, 
            strokes_perpendicular, 
            loop_neighboring_vertical, 
            loop_neighboring_horizontal, 
            loop_neighboring_contained,
            stroke_to_loop,
            stroke_to_edge
        )
        
        
        # 5) If it satisfy the condition, we can build the operations
        if gnn_graph._full_shape and gnn_graph._has_circle_shape():

            print("build !!")

            # 5.1) Do sketch
            sketch_selection_mask, sketch_points, normal = do_sketch(gnn_graph)
            cur__brep_class._sketch_op(sketch_points, normal, sketch_points)

            print("sketch_points", sketch_points)


            # 5.2) Do Extrude
            extrude_amount, extrude_direction = do_extrude(gnn_graph, sketch_selection_mask, sketch_points, brep_edges)
            print("extrude_direction", extrude_direction)

            cur__brep_class.extrude_op(abs(extrude_amount), extrude_direction)


            # 5.3) Write to brep
            cur__brep_class.write_to_json(output_dir)


            # 5.4) Read the program and produce the brep file
            if os.path.exists(output_relative_dir):
                shutil.rmtree(output_relative_dir)
            os.makedirs(output_relative_dir, exist_ok=True)

            parsed_program_class = Preprocessing.proc_CAD.Program_to_STL.parsed_program(file_path, output_dir)
            parsed_program_class.read_json_file()


            # 5.5) Read brep file
            brep_files = [file_name for file_name in os.listdir(os.path.join(output_dir, 'canvas'))
                    if file_name.startswith('brep_') and file_name.endswith('.step')]
            brep_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))


            # 5.6) Update brep data
            brep_edges, brep_loops = cascade_brep(brep_files)
            

        # n) Lastly, update the strokes 
        stroke_in_graph += 1
        print('------')
