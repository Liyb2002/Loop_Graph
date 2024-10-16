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

import torch
import torch.nn as nn
import torch.optim as optim
import os
import shutil
import numpy as np
import random

# --------------------- Dataset --------------------- #
dataset = Preprocessing.dataloader.Program_Graph_Dataset('dataset/messy_order_eval')
data_loader = DataLoader(dataset, batch_size=1, shuffle=True)


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

    Encoders.helper.vis_selected_loops(gnn_graph['stroke'].x.numpy(), gnn_graph['stroke', 'represents', 'loop'].edge_index, torch.argmax(sketch_selection_mask))

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
    extrude_stroke_idx =  (extrude_selection_mask >= 0.5).nonzero(as_tuple=True)[0]
    
    Encoders.helper.vis_selected_strokes(gnn_graph['stroke'].x.cpu().numpy(), extrude_stroke_idx)
    return extrude_selection_mask

# This extrude_amount, extrude_direction is not total correct. Work on it later
def do_extrude(gnn_graph, sketch_selection_mask, sketch_points, brep_edges):
    extrude_selection_mask = predict_extrude(gnn_graph, sketch_selection_mask)
    extrude_amount, extrude_direction = whole_process_helper.helper.get_extrude_amount(gnn_graph, extrude_selection_mask, sketch_points, brep_edges)
    
    print("extrude_amount", extrude_amount)
    return extrude_amount, extrude_direction



# --------------------- Program Prediction Network --------------------- #
program_graph_encoder = Encoders.gnn.gnn.SemanticModule()
program_graph_decoder_stroke = Encoders.gnn.gnn.Program_Decoder('stroke')
program_graph_decoder_loop = Encoders.gnn.gnn.Program_Decoder('loop')
program_dir = os.path.join(current_dir, 'checkpoints', 'program_prediction')
program_graph_encoder.eval()
program_graph_decoder_stroke.eval()
program_graph_decoder_loop.eval()
program_graph_encoder.load_state_dict(torch.load(os.path.join(program_dir, 'graph_encoder.pth')))
program_graph_decoder_stroke.load_state_dict(torch.load(os.path.join(program_dir, 'graph_decoder.pth')))
program_graph_decoder_loop.load_state_dict(torch.load(os.path.join(program_dir, 'graph_decoder.pth')))


def program_prediction(gnn_graph, past_programs):
    past_programs = whole_process_helper.helper.padd_program(past_programs)
    x_dict = program_graph_encoder(gnn_graph.x_dict, gnn_graph.edge_index_dict)
    output_loop = program_graph_decoder_loop(x_dict, past_programs)
    output_stroke = program_graph_decoder_loop(x_dict, past_programs)
    predicted_class = torch.argmax(output_stroke + output_loop, dim=1)

    return predicted_class


# --------------------- Cascade Brep Features --------------------- #
def cascade_brep(brep_files):
    final_brep_edges = []
    final_cylinder_features = []

    for file_name in brep_files:
        brep_directory = os.path.join(output_dir, 'canvas')
        brep_file_path = os.path.join(brep_directory, file_name)

        edge_features_list, cylinder_features = Preprocessing.SBGCN.brep_read.create_graph_from_step_file(brep_file_path)
        
        if len(final_brep_edges) == 0:
            final_brep_edges = edge_features_list
            final_cylinder_features = cylinder_features
        else:
            # We already have brep
            new_features = Preprocessing.generate_dataset.find_new_features(final_brep_edges, edge_features_list) 
            final_brep_edges += new_features
            final_cylinder_features += cylinder_features

    output_brep_edges = Preprocessing.proc_CAD.helper.pad_brep_features(final_brep_edges + final_cylinder_features)
    brep_loops = Preprocessing.proc_CAD.helper.face_aggregate_networkx(output_brep_edges) + Preprocessing.proc_CAD.helper.face_aggregate_circle_brep(output_brep_edges)
    brep_loops = [list(loop) for loop in brep_loops]

    return output_brep_edges, brep_loops



# --------------------- Main Code --------------------- #
for data in tqdm(data_loader, desc="Generating CAD Programs"):
    _, _, _, stroke_node_features, _, _, _, _, _,_, _, _ = data
    
    stroke_node_features = stroke_node_features.squeeze(0)
    stroke_node_features = stroke_node_features.cpu().numpy()
    stroke_node_features = np.round(stroke_node_features, 4)

    print("NEW SHAPE -----------------!")
    # We only want to process complicated shapes
    if stroke_node_features.shape[0] > 50:
        continue
    
    # Init Brep
    brep_edges = torch.zeros(0)
    brep_loops = []
    file_path = os.path.join(output_dir, 'Program.json')


    # Program State Init
    cur__brep_class = Preprocessing.proc_CAD.generate_program.Brep()
    cur_program = torch.tensor([], dtype=torch.int64)


    # Strokes / Loops in the Graph
    loops_fset = Preprocessing.proc_CAD.helper.face_aggregate_networkx(stroke_node_features) + Preprocessing.proc_CAD.helper.face_aggregate_circle(stroke_node_features)
    stroke_cloud_loops = [list(fset) for fset in loops_fset]
    
    connected_stroke_nodes = Preprocessing.proc_CAD.helper.connected_strokes(stroke_node_features)
    strokes_perpendicular, strokes_non_perpendicular =  Preprocessing.proc_CAD.helper.stroke_relations(stroke_node_features, connected_stroke_nodes)

    loop_neighboring_all = Preprocessing.proc_CAD.helper.loop_neighboring_simple(stroke_cloud_loops)
    loop_neighboring_vertical = Preprocessing.proc_CAD.helper.loop_neighboring_complex(stroke_cloud_loops, stroke_node_features, loop_neighboring_all)
    loop_neighboring_horizontal = Preprocessing.proc_CAD.helper.coplanr_neighorbing_loop(loop_neighboring_all, loop_neighboring_vertical)
    loop_neighboring_contained = Preprocessing.proc_CAD.helper.loop_contained(stroke_cloud_loops, stroke_node_features)


    # Operation prediction
    current_op = 1
    past_programs = []

    while current_op != 0:
    
    # -------------------- Prepare the graph informations -------------------- #
        # 1) Stroke to brep
        stroke_to_loop_lines = Preprocessing.proc_CAD.helper.stroke_to_brep(stroke_cloud_loops, brep_loops, stroke_node_features, brep_edges)
        stroke_to_loop_circle = Preprocessing.proc_CAD.helper.stroke_to_brep_circle(stroke_cloud_loops, brep_loops, stroke_node_features, brep_edges)
        stroke_to_loop = Preprocessing.proc_CAD.helper.union_matrices(stroke_to_loop_lines, stroke_to_loop_circle)
        
        stroke_to_edge_lines = Preprocessing.proc_CAD.helper.stroke_to_edge(stroke_node_features, brep_edges)
        stroke_to_edge_circle = Preprocessing.proc_CAD.helper.stroke_to_edge_circle(stroke_node_features, brep_edges)
        stroke_to_edge = Preprocessing.proc_CAD.helper.union_matrices(stroke_to_edge_lines, stroke_to_edge_circle)
        
        # 2) Build graph
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
        
        # Encoders.helper.vis_left_graph(gnn_graph['stroke'].x.cpu().numpy())

        

        # 3) Build operations

        # 3.1) Do sketch
        sketch_selection_mask, sketch_points, normal = do_sketch(gnn_graph)
        if sketch_points.shape[0] == 1:
            # do circle sketch
            cur__brep_class.regular_sketch_circle(sketch_points[0, 3:6].tolist(), sketch_points[0, 7].item(), sketch_points[0, :3].tolist())
        else: 
            cur__brep_class._sketch_op(sketch_points, normal, sketch_points)


        # 3.2) Do Extrude
        extrude_amount, extrude_direction = do_extrude(gnn_graph, sketch_selection_mask, sketch_points, brep_edges)
        cur__brep_class.extrude_op(extrude_amount, extrude_direction)


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
        Encoders.helper.vis_brep(brep_edges)
        
        past_programs.append(1)
        past_programs.append(2)
        current_op = program_prediction(gnn_graph, past_programs)


