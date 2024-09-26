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
dataset = Preprocessing.dataloader.Program_Graph_Dataset('dataset/eval')
good_data_indices = [i for i, data in enumerate(dataset) if data[0][-1] == 0]
filtered_dataset = Subset(dataset, good_data_indices)
data_loader = DataLoader(dataset, batch_size=1, shuffle=True)


# --------------------- Directory --------------------- #
current_dir = os.getcwd()
output_dir = os.path.join(current_dir, 'program_output')
output_relative_dir = ('program_output/canvas')



# --------------------- Skecth Network --------------------- #
def do_sketch(node_features, face_to_stroke, gnn_strokeCloud_edges, stroke_cloud_coplanar, cur_brep_stroke_connection):
    # Load models
    loop_embed_model = Models.loop_embeddings.LoopEmbeddingNetwork()
    graph_encoder = Encoders.gnn.gnn.SemanticModule()
    graph_decoder = Encoders.gnn.gnn.Sketch_brep_prediction_timeEmbed()
    loop_embed_model.to(device)
    graph_encoder.to(device)
    graph_decoder.to(device)
    dir = os.path.join(current_dir, 'checkpoints', 'sketch_prediction_timeEmbed')
    loop_embed_model.load_state_dict(torch.load(os.path.join(dir, 'loop_embed_model.pth')))
    graph_encoder.load_state_dict(torch.load(os.path.join(dir, 'graph_encoder.pth')))
    graph_decoder.load_state_dict(torch.load(os.path.join(dir, 'graph_decoder.pth')))

    # Prepare the graph
    sketch_loop_embeddings = loop_embed_model(node_features, face_to_stroke)
    
    # Build graph
    gnn_graph = Preprocessing.gnn_graph.SketchHeteroData(sketch_loop_embeddings, gnn_strokeCloud_edges, stroke_cloud_coplanar, cur_brep_stroke_connection)
    x_dict = graph_encoder(gnn_graph.x_dict, gnn_graph.edge_index_dict)

    # Predict
    prediction = graph_decoder(x_dict, face_to_stroke).squeeze(0)
    predicted_face_index = torch.argmax(prediction).item()
    choice_tensor = torch.zeros_like(prediction, dtype=torch.int)
    choice_tensor[predicted_face_index] = 1

    # Output
    sketch_strokes = Preprocessing.proc_CAD.helper.get_sketch_points(predicted_face_index, face_to_stroke, node_features)
    sketch_points = Preprocessing.proc_CAD.helper.extract_unique_points(sketch_strokes)
    normal = [1, 0, 0]

    return choice_tensor, sketch_points, normal



# --------------------- Extrude Network --------------------- #
def do_extrude(node_features, face_to_stroke, edge_features, face_choice):
    # Load models
    graph_encoder = Encoders.gnn_stroke.gnn.SemanticModule()
    graph_decoder = Encoders.gnn_stroke.gnn.ExtrudingStrokePrediction()
    graph_encoder.to(device)
    graph_decoder.to(device)
    dir = os.path.join(current_dir, 'checkpoints', 'extrude_stroke')
    graph_encoder.load_state_dict(torch.load(os.path.join(dir, 'graph_encoder.pth')))
    graph_decoder.load_state_dict(torch.load(os.path.join(dir, 'graph_decoder.pth')))

    # Prev sketch
    chosen_face_index = torch.argmax(face_choice).item()
    chosen_strokes = face_to_stroke[chosen_face_index]
    num_strokes = node_features.shape[0]
    stroke_choice = torch.zeros((num_strokes, 1), dtype=torch.int)
    for stroke_tensor in chosen_strokes:
        stroke_index = stroke_tensor.item()
        stroke_choice[stroke_index] = 1

    # Create graph
    intersection_matrix = Encoders.helper.build_intersection_matrix(node_features).to(torch.float32).to(device)
    gnn_graph = Preprocessing.gnn_graph_stroke.SketchHeteroData(node_features, intersection_matrix)
    gnn_graph.set_brep_connection(edge_features)

    # Forward pass
    x_dict = graph_encoder(gnn_graph.x_dict, gnn_graph.edge_index_dict)
    output = graph_decoder(x_dict, gnn_graph.edge_index_dict, stroke_choice)

    # Get values of extrude_amount, direction
    extrude_amount, direction = Preprocessing.proc_CAD.helper.get_extrude_amount(node_features, output, stroke_choice, gnn_graph.x_dict['brep'])
    return extrude_amount, direction




# --------------------- Cascade Brep Features --------------------- #

def cascade_brep(brep_files, node_features):
    final_brep_edges = []
    prev_brep_edges = []
    
    for file_name in brep_files:
        brep_directory = os.path.join(output_dir, 'canvas')
        brep_file_path = os.path.join(brep_directory, file_name)

        edge_features_list, edge_coplanar_list= Preprocessing.SBGCN.brep_read.create_graph_from_step_file(brep_file_path)
        if len(prev_brep_edges) == 0:
            final_brep_edges = edge_features_list
            prev_brep_edges = edge_features_list
        else:
            # We already have brep
            new_features= Preprocessing.generate_dataset.find_new_features(prev_brep_edges, edge_features_list) 
            final_brep_edges += new_features
            prev_brep_edges = edge_features_list

    brep_to_stroke = Preprocessing.proc_CAD.helper.face_to_stroke(final_brep_edges)
    node_features = node_features.numpy()
    brep_stroke_connection = Preprocessing.proc_CAD.helper.stroke_to_brep(face_to_stroke, brep_to_stroke, node_features, final_brep_edges)
    # print("final_brep_edges", final_brep_edges)
    # print("node_features", node_features)
    # print('face_to_stroke', face_to_stroke)
    # print("brep_to_stroke", brep_to_stroke)
    # print("brep_stroke_connection", brep_stroke_connection)
    return brep_stroke_connection

# --------------------- Main Code --------------------- #
for batch in tqdm(data_loader):
    program, node_features, operations_order_matrix, gnn_strokeCloud_edges, face_to_stroke, stroke_cloud_coplanar, brep_to_stroke, edge_features, gnn_brep_edges, brep_stroke_connection, brep_coplanar = Preprocessing.dataloader.process_batch(batch)

    face_to_stroke_transformed = []
    for sublist in face_to_stroke:
        transformed_sublist = [int(tensor.item()) for tensor in sublist]
        face_to_stroke_transformed.append(transformed_sublist)

    # Parser Init
    file_path = os.path.join(output_dir, 'Program.json')

    # Program State init
    cur__brep_class = Preprocessing.proc_CAD.generate_program.Brep()
    cur_program = torch.tensor([], dtype=torch.int64)
    cur_brep_stroke_connection = torch.zeros((2, 0), dtype=torch.float32)

    # Graph init
    next_op = 1
    
    while next_op != 0:
        print("Op Executing", next_op)


        # Terminate
        if next_op == 0:
            break
            
        # Sketch
        if next_op == 1:
            prev_sketch_matrix, sketch_points, normal = do_sketch(node_features, face_to_stroke, gnn_strokeCloud_edges, stroke_cloud_coplanar, cur_brep_stroke_connection)
            cur__brep_class._sketch_op(sketch_points, normal, sketch_points.tolist())
            Encoders.helper.vis_gt(prev_sketch_matrix, face_to_stroke, node_features)
            

        # Extrude
        if next_op == 2:
            extrude_amount, direction = do_extrude(node_features, face_to_stroke, edge_features, prev_sketch_matrix)
            print("direction", direction.tolist())
            cur__brep_class.extrude_op(extrude_amount, direction.tolist())
        
        # Write the Program
        cur__brep_class.write_to_json(output_dir)

        # Read the Program and produce brep file
        if os.path.exists(output_relative_dir):
            shutil.rmtree(output_relative_dir)
        os.makedirs(output_relative_dir, exist_ok=True)

        parsed_program_class = Preprocessing.proc_CAD.Program_to_STL.parsed_program(file_path, output_dir)
        parsed_program_class.read_json_file()

        # Read brep file
        brep_files = [file_name for file_name in os.listdir(os.path.join(output_dir, 'canvas'))
                if file_name.startswith('brep_') and file_name.endswith('.step')]
        brep_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
        brep_file_path = brep_files[-1]
        brep_file_path = os.path.join(output_relative_dir, brep_file_path)
        

        # Update brep data
        print("brep_file_path", brep_file_path)
        cur_brep_stroke_connection = cascade_brep(brep_files, node_features)
        cur_brep_stroke_connection = torch.from_numpy(cur_brep_stroke_connection)
        edge_features_list, edge_coplanar_list= Preprocessing.SBGCN.brep_read.create_graph_from_step_file(brep_file_path) 
        edge_features = torch.tensor(edge_features_list)
        print("edge_features", edge_features.shape)
        Encoders.helper.vis_stroke_cloud(edge_features)

        # Predict next Operation
        if next_op == 1:
            next_op = 2
        else:
            next_op = 1

    break