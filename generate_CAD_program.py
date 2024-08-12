import Preprocessing.dataloader
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


# --------------------- Directory --------------------- #
def brep_info_to_graph_info():
    edge_features = Preprocessing.proc_CAD.helper.preprocess_features(edge_features_list)

    brep_to_stroke = Preprocessing.proc_CAD.helper.brep_to_stroke(face_feature_gnn_list, edge_features)
    gnn_brep_edges = Preprocessing.proc_CAD.helper.gnn_edges(brep_to_stroke)

    brep_stroke_connection = Preprocessing.proc_CAD.helper.stroke_to_brep(face_to_stroke, brep_to_stroke, node_features, edge_features)
    brep_coplanar = Preprocessing.proc_CAD.helper.coplanar_matrix(brep_to_stroke, edge_features)

    return edge_features, brep_to_stroke, gnn_brep_edges, brep_stroke_connection, brep_coplanar

# --------------------- Skecth Network --------------------- #
def do_sketch(node_features, face_to_stroke, edge_features, brep_to_stroke, gnn_strokeCloud_edges, gnn_brep_edges, brep_stroke_connection, stroke_cloud_coplanar, brep_coplanar):
    # Load models
    loop_embed_model = Models.loop_embeddings.LoopEmbeddingNetwork()
    graph_encoder = Encoders.gnn.gnn.SemanticModule()
    graph_decoder = Encoders.gnn.gnn.Sketch_brep_prediction()
    loop_embed_model.to(device)
    graph_encoder.to(device)
    graph_decoder.to(device)
    dir = os.path.join(current_dir, 'checkpoints', 'sketch_prediction')
    loop_embed_model.load_state_dict(torch.load(os.path.join(dir, 'loop_embed_model.pth')))
    graph_encoder.load_state_dict(torch.load(os.path.join(dir, 'graph_encoder.pth')))
    graph_decoder.load_state_dict(torch.load(os.path.join(dir, 'graph_decoder.pth')))

    # Prepare the graph
    sketch_loop_embeddings = loop_embed_model(node_features, face_to_stroke)
    brep_loop_embeddings = loop_embed_model(edge_features, brep_to_stroke)
    
    # Build graph
    gnn_graph = Preprocessing.gnn_graph.SketchHeteroData(sketch_loop_embeddings, brep_loop_embeddings, gnn_strokeCloud_edges, gnn_brep_edges, brep_stroke_connection, stroke_cloud_coplanar, brep_coplanar)
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
def do_extrude(node_features, face_to_stroke, edge_features, brep_to_stroke, gnn_strokeCloud_edges, gnn_brep_edges, brep_stroke_connection, stroke_cloud_coplanar, brep_coplanar, face_choice):
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

    print("stroke_choice", stroke_choice)



# --------------------- Program Prediction Network --------------------- #
def do_ProgramPredict(node_features, face_to_stroke, edge_features, brep_to_stroke, gnn_strokeCloud_edges, gnn_brep_edges, brep_stroke_connection, stroke_cloud_coplanar, brep_coplanar, cur_program):
    # Load models
    loop_embed_model = Models.loop_embeddings.LoopEmbeddingNetwork()
    graph_encoder = Encoders.gnn.gnn.SemanticModule()
    graph_decoder = Encoders.gnn.gnn.Program_prediction()
    loop_embed_model.to(device)
    graph_encoder.to(device)
    graph_decoder.to(device)
    dir = os.path.join(current_dir, 'checkpoints', 'program_prediction')
    loop_embed_model.load_state_dict(torch.load(os.path.join(dir, 'loop_embed_model.pth')))
    graph_encoder.load_state_dict(torch.load(os.path.join(dir, 'graph_encoder.pth')))
    graph_decoder.load_state_dict(torch.load(os.path.join(dir, 'graph_decoder.pth')))

    # Prepare the graph
    sketch_loop_embeddings = loop_embed_model(node_features, face_to_stroke)
    brep_loop_embeddings = loop_embed_model(edge_features, brep_to_stroke)

    # Build graph
    gnn_graph = Preprocessing.gnn_graph.SketchHeteroData(sketch_loop_embeddings, brep_loop_embeddings, gnn_strokeCloud_edges, gnn_brep_edges, brep_stroke_connection, stroke_cloud_coplanar, brep_coplanar)
    x_dict = graph_encoder(gnn_graph.x_dict, gnn_graph.edge_index_dict)

    # Predict
    prediction = graph_decoder(x_dict, cur_program).squeeze(0)
    predicted_token = torch.argmax(prediction)
    return predicted_token


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

    cur_brep_to_stroke = torch.zeros((1,0), dtype=torch.float32)
    cur_edge_features = torch.zeros((1,0), dtype=torch.float32)
    cur_gnn_brep_edges = torch.zeros((2, 0), dtype=torch.float32)
    cur_brep_stroke_connection = torch.zeros((2, 0), dtype=torch.float32)
    cur_brep_coplanar = torch.zeros((2, 0), dtype=torch.float32)

    # Graph init
    next_op = 1

    while next_op != 0:
        print("Op Executing", next_op)

        # Terminate
        if next_op == 0:
            break
            
        # Sketch
        if next_op == 1:
            prev_sketch_matrix, sketch_points, normal = do_sketch(node_features, face_to_stroke, cur_edge_features, cur_brep_to_stroke, gnn_strokeCloud_edges, cur_gnn_brep_edges, cur_brep_stroke_connection, stroke_cloud_coplanar, cur_brep_coplanar)
            cur__brep_class._sketch_op(sketch_points, normal, sketch_points.tolist())

        # Extrude
        if next_op == 2:
            do_extrude(node_features, face_to_stroke, edge_features, brep_to_stroke, gnn_strokeCloud_edges, gnn_brep_edges, brep_stroke_connection, stroke_cloud_coplanar, brep_coplanar, prev_sketch_matrix)

        
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
        face_feature_gnn_list, face_features_list, edge_features_list, vertex_features_list, edge_index_face_edge_list, edge_index_edge_vertex_list, edge_index_face_face_list, index_id= Preprocessing.SBGCN.brep_read.create_graph_from_step_file(brep_file_path)
        cur_edge_features = Preprocessing.proc_CAD.helper.preprocess_features(edge_features_list)
        cur_brep_to_stroke = Preprocessing.proc_CAD.helper.brep_to_stroke(face_feature_gnn_list, cur_edge_features)        
        cur_gnn_brep_edges = Preprocessing.proc_CAD.helper.gnn_edges(cur_brep_to_stroke)
        cur_brep_stroke_connection = Preprocessing.proc_CAD.helper.stroke_to_brep(face_to_stroke_transformed, cur_brep_to_stroke, node_features, cur_edge_features)
        cur_brep_coplanar = Preprocessing.proc_CAD.helper.coplanar_matrix(cur_brep_to_stroke, cur_edge_features)

        cur_edge_features = torch.tensor(cur_edge_features, dtype=torch.float32)
        cur_brep_to_stroke = [[torch.tensor([value], dtype=torch.int) for value in sublist] for sublist in cur_brep_to_stroke]
        cur_gnn_brep_edges = torch.tensor(cur_gnn_brep_edges, dtype=torch.float32)
        cur_brep_stroke_connection = torch.tensor(cur_brep_stroke_connection, dtype=torch.float32)
        cur_brep_coplanar = torch.tensor(cur_brep_coplanar, dtype=torch.float32)

        # Predict next Operation
        cur_program = torch.cat((cur_program, torch.tensor([next_op], dtype=torch.int64)))
        next_op = do_ProgramPredict(node_features, face_to_stroke, cur_edge_features, cur_brep_to_stroke, gnn_strokeCloud_edges, cur_gnn_brep_edges, cur_brep_stroke_connection, stroke_cloud_coplanar, cur_brep_coplanar, cur_program)
        print("------------")