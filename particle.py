import Preprocessing.dataloader
import Preprocessing.generate_dataset_baseline
import Preprocessing.gnn_graph

import Preprocessing.proc_CAD.Program_to_STL
import Preprocessing.proc_CAD.brep_read
import Preprocessing.proc_CAD.helper

import whole_process_helper.helper

import Models.loop_embeddings

import Encoders.gnn.gnn
import Encoders.gnn_stroke.gnn
import Encoders.helper

from Preprocessing.config import device

from torch.utils.data import DataLoader
from tqdm import tqdm

import copy
import json

import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import os
import shutil
import numpy as np
import random


class Particle():
    def __init__(self, gt_brep_file_path, data_produced, stroke_node_features):
        

        print("new particle!")

        stroke_node_features = np.round(stroke_node_features, 4)
        self.stroke_node_features = stroke_node_features
        

        self.gt_brep_file_path = gt_brep_file_path
        self.get_gt_brep_history()


        self.data_produced = data_produced

        self.brep_edges = torch.zeros(0)
        self.brep_loops = []
        self.cur__brep_class = Preprocessing.proc_CAD.generate_program.Brep()


        loops_fset = Preprocessing.proc_CAD.helper.face_aggregate_networkx(stroke_node_features) + Preprocessing.proc_CAD.helper.face_aggregate_circle(stroke_node_features)
        self.stroke_cloud_loops = [list(fset) for fset in loops_fset]
        
        self.connected_stroke_nodes = Preprocessing.proc_CAD.helper.connected_strokes(stroke_node_features)
        self.strokes_perpendicular, strokes_non_perpendicular =  Preprocessing.proc_CAD.helper.stroke_relations(stroke_node_features, self.connected_stroke_nodes)

        self.loop_neighboring_all = Preprocessing.proc_CAD.helper.loop_neighboring_simple(self.stroke_cloud_loops)
        self.loop_neighboring_vertical = Preprocessing.proc_CAD.helper.loop_neighboring_complex(self.stroke_cloud_loops, self.stroke_node_features, self.loop_neighboring_all)
        self.loop_neighboring_horizontal = Preprocessing.proc_CAD.helper.coplanr_neighorbing_loop(self.loop_neighboring_all, self.loop_neighboring_vertical)
        self.loop_neighboring_contained = Preprocessing.proc_CAD.helper.loop_contained(self.stroke_cloud_loops, stroke_node_features)

        self.current_op = 1
        self.past_programs = [9]


        # Iteration infos
        self.selected_loop_indices = []

        # Particle State
        self.valid_particle = True
        self.success_terminate = False
        self.score = 1

        # Feature_strokes
        self.predicted_feature_strokes = None



    def set_particle_id(self, particle_id, cur_output_dir_outerFolder):
        self.cur_output_dir = os.path.join(cur_output_dir_outerFolder, f'particle_{particle_id}')
        os.makedirs(self.cur_output_dir, exist_ok=True)
        
        self.particle_id = particle_id
        self.file_path = os.path.join(self.cur_output_dir, 'Program.json')


    def deepcopy_particle(self, new_id):

        new_particle = copy.copy(self)
        
        # manual copy, because we have tensors
        new_particle.brep_edges = self.brep_edges.clone()
        new_particle.brep_loops = self.brep_loops[:]
        new_particle.cur__brep_class = copy.deepcopy(self.cur__brep_class)
        new_particle.stroke_cloud_loops = copy.deepcopy(self.stroke_cloud_loops)
        new_particle.connected_stroke_nodes = copy.deepcopy(self.connected_stroke_nodes)
        new_particle.strokes_perpendicular = copy.deepcopy(self.strokes_perpendicular)
        new_particle.loop_neighboring_all = copy.deepcopy(self.loop_neighboring_all)
        new_particle.loop_neighboring_vertical = copy.deepcopy(self.loop_neighboring_vertical)
        new_particle.loop_neighboring_horizontal = copy.deepcopy(self.loop_neighboring_horizontal)
        new_particle.loop_neighboring_contained = copy.deepcopy(self.loop_neighboring_contained)
        new_particle.current_op = self.current_op
        new_particle.past_programs = self.past_programs[:]
        new_particle.selected_loop_indices = self.selected_loop_indices[:]
        new_particle.valid_particle = self.valid_particle
        new_particle.success_terminate = self.success_terminate
        new_particle.score = self.score
        new_particle.predicted_feature_strokes = (self.predicted_feature_strokes.clone() if self.predicted_feature_strokes is not None else None)
        new_particle.gt_program = self.gt_program


        cur_output_dir_outerFolder = os.path.dirname(self.cur_output_dir)
        new_folder_path = os.path.join(cur_output_dir_outerFolder, f'particle_{new_id}')
        shutil.copytree(self.cur_output_dir, new_folder_path)

        new_particle.particle_id = new_id
        new_particle.cur_output_dir = new_folder_path
        new_particle.file_path = os.path.join(new_folder_path, 'Program.json')
        new_particle.valid_particle = True

        return new_particle


    def set_gt_program(self, program):
        self.gt_program = program


    def program_terminated(self, gnn_graph):

        # if self.predicted_feature_strokes is None:
        #     return False
        
        # termination_prob, untouched_feature_idx= whole_process_helper.helper.sample_program_termination(gnn_graph['stroke'].x.cpu().numpy(), self.predicted_feature_strokes)
        # # Encoders.helper.vis_selected_strokes(gnn_graph['stroke'].x.cpu().numpy(), [])

        # if random.random() < termination_prob or len(self.past_programs) > 20: 
        #     return True
        
        if (len(self.gt_program) == len(self.past_programs)):
            stroke_features_file = os.path.join(self.cur_output_dir, 'stroke_cloud_features.json')
            stroke_features_list = self.stroke_node_features.tolist()

            with open(stroke_features_file, 'w') as json_file:
                for stroke in stroke_features_list:
                    json.dump(stroke, json_file)
                    json_file.write("\n")

        return len(self.gt_program) == len(self.past_programs)
        


    def particle_score(self):
        return self.score
    

    def is_valid_particle(self):
        return self.valid_particle


    def generate_next_step(self):

        try:

            stroke_to_loop_lines = Preprocessing.proc_CAD.helper.stroke_to_brep(self.stroke_cloud_loops, self.brep_loops, self.stroke_node_features, self.brep_edges)
            stroke_to_loop_circle = Preprocessing.proc_CAD.helper.stroke_to_brep_circle(self.stroke_cloud_loops, self.brep_loops, self.stroke_node_features, self.brep_edges)
            stroke_to_loop = Preprocessing.proc_CAD.helper.union_matrices(stroke_to_loop_lines, stroke_to_loop_circle)
            

            stroke_to_edge_lines = Preprocessing.proc_CAD.helper.stroke_to_edge(self.stroke_node_features, self.brep_edges)
            stroke_to_edge_circle = Preprocessing.proc_CAD.helper.stroke_to_edge_circle(self.stroke_node_features, self.brep_edges)
            stroke_to_edge = Preprocessing.proc_CAD.helper.union_matrices(stroke_to_edge_lines, stroke_to_edge_circle)
            
            # 2) Build graph
            gnn_graph = Preprocessing.gnn_graph.SketchLoopGraph(
                self.stroke_cloud_loops, 
                self.stroke_node_features, 
                self.strokes_perpendicular, 
                self.loop_neighboring_vertical, 
                self.loop_neighboring_horizontal, 
                self.loop_neighboring_contained,
                stroke_to_loop,
                stroke_to_edge
            )

            # Encoders.helper.vis_left_graph(gnn_graph['stroke'].x.cpu().numpy())

            if len(self.past_programs) == 1:
                # Find all feature edges
                self.predicted_feature_strokes = do_stroke_type_prediction(gnn_graph)


            if self.current_op == 1:
                # print("Build sketch")
                self.sketch_selection_mask, self.sketch_points, normal, selected_loop_idx, prob = do_sketch(gnn_graph)
                self.selected_loop_indices.append(selected_loop_idx)
                self.score = self.score * prob
                if self.sketch_points.shape[0] == 1:
                    # do circle sketch
                    self.cur__brep_class.regular_sketch_circle(self.sketch_points[0, 3:6].tolist(), self.sketch_points[0, 7].item(), self.sketch_points[0, :3].tolist())
                else: 
                    self.cur__brep_class._sketch_op(self.sketch_points, normal, self.sketch_points)


            # Build Extrude
            if self.current_op == 2:
                # print("Build extrude")
                extrude_target_point, prob = do_extrude(gnn_graph, self.sketch_selection_mask, self.sketch_points, self.brep_edges)
                self.cur__brep_class.extrude_op(extrude_target_point)
                self.score = self.score * prob


            # Build fillet
            if self.current_op == 3:
                # print("Build Fillet")
                fillet_edge, fillet_amount, prob = do_fillet(gnn_graph, self.brep_edges)
                self.cur__brep_class.random_fillet(fillet_edge, fillet_amount)
                self.score = self.score * prob


            if self.current_op ==4:
                print("Build Chamfer")
                chamfer_edge, chamfer_amount, prob= do_chamfer(gnn_graph, self.brep_edges)
                self.cur__brep_class.random_chamfer(chamfer_edge, chamfer_amount)
                self.score = self.score * prob


            # 5.3) Write to brep
            self.cur__brep_class.write_to_json(self.cur_output_dir)


            # 5.4) Read the program and produce the brep file
            parsed_program_class = Preprocessing.proc_CAD.Program_to_STL.parsed_program(self.file_path, self.cur_output_dir)
            parsed_program_class.read_json_file()


            # 5.5) Read brep file
            cur_relative_output_dir = os.path.join('program_output/', f'data_{self.data_produced}', f'particle_{self.particle_id}')

            brep_files = [file_name for file_name in os.listdir(os.path.join(cur_relative_output_dir, 'canvas'))
                    if file_name.startswith('brep_') and file_name.endswith('.step')]
            brep_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))


            # 5.6) Update brep data
            prev_brep_edges = self.brep_edges
            brep_path = os.path.join('program_output/', f'data_{self.data_produced}', f'particle_{self.particle_id}', 'canvas')
            self.brep_edges, self.brep_loops, new_edges_in_current_step = cascade_brep_accumulate(self.brep_edges, self.brep_loops, brep_files, self.data_produced, brep_path)
            # Encoders.helper.vis_brep(self.brep_edges)

            # Check if extrude op really adds something new
            if self.current_op == 2 and len(self.past_programs) > 2:
                if prev_brep_edges.shape[0] == self.brep_edges.shape[0]:
                    self.valid_particle = False
                    return 


            # Compute Chamfer Distance
            on_right_track, high_dist_indices= chamfer_distance_brep(self.gt_brep_edges, new_edges_in_current_step)


            self.past_programs.append(self.current_op)
            self.current_op, op_prob = program_prediction(gnn_graph, self.past_programs)
            self.score = self.score * op_prob


            print("----------------")
            print("self.past_programs", self.past_programs)
            print("self.gt_program", self.gt_program)
            print("self.current_op", self.current_op)

            # 6) Write the stroke_cloud data to pkl file
            output_file_path = os.path.join(self.cur_output_dir, 'canvas', f'{len(brep_files)-1}_eval_info.pkl')
            with open(output_file_path, 'wb') as f:
                pickle.dump({
                    'stroke_node_features': self.stroke_node_features,
                    'gt_brep_edges': self.gt_brep_edges,
                    'on_right_track' : on_right_track, 
                    'high_dist_indices': high_dist_indices
                }, f)
            

            # 7) Also copy the gt brep file
            gt_brep_folder = os.path.dirname(self.gt_brep_file_path)

            # Create a new folder called gt_canvas in self.cur_output_dir
            gt_canvas_dir = os.path.join(self.cur_output_dir, 'gt_canvas')
            os.makedirs(gt_canvas_dir, exist_ok=True)

            # Copy everything from gt_brep_folder to gt_canvas
            for item in os.listdir(gt_brep_folder):
                source_item = os.path.join(gt_brep_folder, item)
                dest_item = os.path.join(gt_canvas_dir, item)
                
                if os.path.isdir(source_item):
                    shutil.copytree(source_item, dest_item, dirs_exist_ok=True)
                else:
                    shutil.copy2(source_item, dest_item)
            
            shutil.copy(self.gt_brep_file_path, os.path.join(self.cur_output_dir, 'gt_brep.step'))

            whole_process_helper.helper.brep_to_stl_and_copy(self.gt_brep_file_path, self.cur_output_dir,os.path.join(self.cur_output_dir, 'gt_brep.step'))

        
        except Exception as e:
            print(f"An error occurred: {e}")
            self.valid_particle = False
    


        if self.program_terminated(gnn_graph):
            self.valid_particle = False
            self.success_terminate = True
            return 


    def get_gt_brep_history(self):
        brep_path = os.path.dirname(self.gt_brep_file_path)
        brep_files = [f for f in os.listdir(brep_path) if f.endswith('.step')]
        
        self.gt_brep_edges, _ = cascade_brep(brep_files, None, brep_path)



    def remove_particle(self):
        shutil.rmtree(self.cur_output_dir)



# ---------------------------------------------------------------------------------------------------------------------------------- #



# --------------------- Directory --------------------- #
current_dir = os.getcwd()
output_dir = os.path.join(current_dir, 'program_output')


# --------------------- Skecth Network --------------------- #
sketch_graph_encoder = Encoders.gnn.gnn.SemanticModule()
sketch_graph_decoder = Encoders.gnn.gnn.Sketch_Decoder()
sketch_graph_encoder.eval()
sketch_graph_decoder.eval()
sketch_dir = os.path.join(current_dir, 'checkpoints', 'sketch_prediction')
sketch_graph_encoder.load_state_dict(torch.load(os.path.join(sketch_dir, 'graph_encoder.pth'), weights_only=True))
sketch_graph_decoder.load_state_dict(torch.load(os.path.join(sketch_dir, 'graph_decoder.pth'), weights_only=True))

def predict_sketch(gnn_graph):
        
    x_dict = sketch_graph_encoder(gnn_graph.x_dict, gnn_graph.edge_index_dict)
    sketch_selection_mask = sketch_graph_decoder(x_dict)

    selected_loop_idx, idx_prob = whole_process_helper.helper.find_valid_sketch(gnn_graph, sketch_selection_mask)
    sketch_stroke_idx = Encoders.helper.find_selected_strokes_from_loops(gnn_graph['stroke', 'represents', 'loop'].edge_index, selected_loop_idx)

    # Encoders.helper.vis_selected_strokes(gnn_graph['stroke'].x.cpu().numpy(), sketch_stroke_idx)

    return selected_loop_idx, sketch_selection_mask, idx_prob

def do_sketch(gnn_graph):
    selected_loop_idx, sketch_selection_mask, idx_prob= predict_sketch(gnn_graph)
    sketch_points = whole_process_helper.helper.extract_unique_points(selected_loop_idx[0], gnn_graph)

    normal = [1, 0, 0]
    sketch_selection_mask = whole_process_helper.helper.clean_mask(sketch_selection_mask, selected_loop_idx)
    return sketch_selection_mask, sketch_points, normal, selected_loop_idx, idx_prob


# --------------------- Extrude Network --------------------- #
extrude_graph_encoder = Encoders.gnn.gnn.SemanticModule()
extrude_graph_decoder = Encoders.gnn.gnn.Extrude_Decoder()
extrude_dir = os.path.join(current_dir, 'checkpoints', 'extrude_prediction')
extrude_graph_encoder.eval()
extrude_graph_decoder.eval()
extrude_graph_encoder.load_state_dict(torch.load(os.path.join(extrude_dir, 'graph_encoder.pth'), weights_only=True))
extrude_graph_decoder.load_state_dict(torch.load(os.path.join(extrude_dir, 'graph_decoder.pth'), weights_only=True))

def predict_extrude(gnn_graph, sketch_selection_mask):
    gnn_graph.set_select_sketch(sketch_selection_mask)

    x_dict = extrude_graph_encoder(gnn_graph.x_dict, gnn_graph.edge_index_dict)
    extrude_selection_mask = extrude_graph_decoder(x_dict)
    
    extrude_stroke_idx =  (extrude_selection_mask >= 0.5).nonzero(as_tuple=True)[0]
    # _, extrude_stroke_idx = torch.max(extrude_selection_mask, dim=0)
    # Encoders.helper.vis_selected_strokes(gnn_graph['stroke'].x.cpu().numpy(), extrude_stroke_idx)
    return extrude_selection_mask

def do_extrude(gnn_graph, sketch_selection_mask, sketch_points, brep_edges):
    extrude_selection_mask = predict_extrude(gnn_graph, sketch_selection_mask)
    extrude_target_point, selected_prob= whole_process_helper.helper.get_extrude_amount(gnn_graph, extrude_selection_mask, sketch_points, brep_edges)

    return extrude_target_point, selected_prob



# --------------------- Fillet Network --------------------- #
fillet_graph_encoder = Encoders.gnn.gnn.SemanticModule()
fillet_graph_decoder = Encoders.gnn.gnn.Fillet_Decoder()
fillet_dir = os.path.join(current_dir, 'checkpoints', 'fillet_prediction')
fillet_graph_encoder.eval()
fillet_graph_decoder.eval()
fillet_graph_encoder.load_state_dict(torch.load(os.path.join(fillet_dir, 'graph_encoder.pth'), weights_only=True))
fillet_graph_decoder.load_state_dict(torch.load(os.path.join(fillet_dir, 'graph_decoder.pth'), weights_only=True))


def predict_fillet(gnn_graph):
    
    x_dict = fillet_graph_encoder(gnn_graph.x_dict, gnn_graph.edge_index_dict)
    fillet_selection_mask = fillet_graph_decoder(x_dict)

    fillet_stroke_idx =  (fillet_selection_mask >= 0.3).nonzero(as_tuple=True)[0]
    # _, fillet_stroke_idx = torch.topk(fillet_selection_mask.flatten(), k=1)
    # _, fillet_stroke_idx = torch.max(fillet_selection_mask, dim=0)

    # Encoders.helper.vis_selected_strokes(gnn_graph['stroke'].x.cpu().numpy(), fillet_stroke_idx)
    return fillet_selection_mask


def do_fillet(gnn_graph, brep_edges):
    fillet_selection_mask = predict_fillet(gnn_graph)
    fillet_edge, fillet_amount, selected_prob= whole_process_helper.helper.get_fillet_amount(gnn_graph, fillet_selection_mask, brep_edges)

    return fillet_edge, fillet_amount.item(), selected_prob





# --------------------- Chamfer Network --------------------- #
chamfer_graph_encoder = Encoders.gnn.gnn.SemanticModule()
chamfer_graph_decoder = Encoders.gnn.gnn.Chamfer_Decoder()
chanfer_dir = os.path.join(current_dir, 'checkpoints', 'chamfer_prediction')
chamfer_graph_encoder.eval()
chamfer_graph_decoder.eval()
chamfer_graph_encoder.load_state_dict(torch.load(os.path.join(chanfer_dir, 'graph_encoder.pth'), weights_only=True))
chamfer_graph_decoder.load_state_dict(torch.load(os.path.join(chanfer_dir, 'graph_decoder.pth'), weights_only=True))


def predict_chamfer(gnn_graph):
    # gnn_graph.padding()
    x_dict = chamfer_graph_encoder(gnn_graph.x_dict, gnn_graph.edge_index_dict)
    chamfer_selection_mask = chamfer_graph_decoder(x_dict)

    # chamfer_stroke_idx =  (chamfer_selection_mask >= 0.3).nonzero(as_tuple=True)[0]
    # _, chamfer_stroke_idx = torch.topk(chamfer_selection_mask.flatten(), k=2)
    _, chamfer_stroke_idx = torch.max(chamfer_selection_mask, dim=0)
    # Encoders.helper.vis_selected_strokes(gnn_graph['stroke'].x.cpu().numpy(), chamfer_stroke_idx)
    
    return chamfer_selection_mask


def do_chamfer(gnn_graph, brep_edges):
    chamfer_selection_mask = predict_chamfer(gnn_graph)
    chamfer_edge, chamfer_amount, selected_prob= whole_process_helper.helper.get_chamfer_amount(gnn_graph, chamfer_selection_mask, brep_edges)
    return chamfer_edge, chamfer_amount.item(), selected_prob




# --------------------- Operation Prediction Network --------------------- #
operation_graph_encoder = Encoders.gnn.gnn.SemanticModule()
operation_graph_decoder= Encoders.gnn.gnn.Program_Decoder()
program_dir = os.path.join(current_dir, 'checkpoints', 'operation_prediction')
operation_graph_encoder.eval()
operation_graph_decoder.eval()
operation_graph_encoder.load_state_dict(torch.load(os.path.join(program_dir, 'graph_encoder.pth'), weights_only=True))
operation_graph_decoder.load_state_dict(torch.load(os.path.join(program_dir, 'graph_decoder.pth'), weights_only=True))


def program_prediction(gnn_graph, past_programs):
    past_programs = whole_process_helper.helper.padd_program(past_programs)
    gnn_graph.padding()
    x_dict = operation_graph_encoder(gnn_graph.x_dict, gnn_graph.edge_index_dict)
    output = operation_graph_decoder(x_dict, past_programs)

    predicted_class, class_prob = whole_process_helper.helper.sample_operation(output)
    return predicted_class, class_prob


# --------------------- Stroke Type Prediction Network --------------------- #
strokeType_graph_encoder = Encoders.gnn.gnn.SemanticModule()
strokeType_graph_decoder= Encoders.gnn.gnn.Stroke_type_Decoder()
strokeType_dir = os.path.join(current_dir, 'checkpoints', 'stroke_type_prediction')
strokeType_graph_encoder.eval()
strokeType_graph_decoder.eval()
strokeType_graph_encoder.load_state_dict(torch.load(os.path.join(strokeType_dir, 'graph_encoder.pth'), weights_only=True))
strokeType_graph_decoder.load_state_dict(torch.load(os.path.join(strokeType_dir, 'graph_decoder.pth'), weights_only=True))


def do_stroke_type_prediction(gnn_graph):
    x_dict = strokeType_graph_encoder(gnn_graph.x_dict, gnn_graph.edge_index_dict)
    output_mask = strokeType_graph_decoder(x_dict)

    predicted_stroke_idx = (output_mask > 0.5).nonzero(as_tuple=True)[0]  # Indices of chosen strokes
    # Encoders.helper.vis_selected_strokes(gnn_graph['stroke'].x.cpu().numpy(), predicted_stroke_idx)
    return output_mask


# --------------------- Cascade Brep Features --------------------- #

def cascade_brep(brep_files, data_produced, brep_path):
    final_brep_edges = []
    final_cylinder_features = []

    for file_name in brep_files:
        brep_file_path = os.path.join(brep_path, file_name)
        edge_features_list, cylinder_features = Preprocessing.SBGCN.brep_read.create_graph_from_step_file(brep_file_path)
        
        
        if len(final_brep_edges) == 0:
            final_brep_edges = edge_features_list
            final_cylinder_features = cylinder_features
        else:
            # We already have brep
            new_features = Preprocessing.generate_dataset_baseline.find_new_features(final_brep_edges, edge_features_list) 
            final_brep_edges += new_features
            final_cylinder_features += cylinder_features

    output_brep_edges = Preprocessing.proc_CAD.helper.pad_brep_features(final_brep_edges + final_cylinder_features)
    brep_loops = Preprocessing.proc_CAD.helper.face_aggregate_networkx(output_brep_edges) + Preprocessing.proc_CAD.helper.face_aggregate_circle_brep(output_brep_edges)
    brep_loops = [list(loop) for loop in brep_loops]

    return output_brep_edges, brep_loops


def cascade_brep_accumulate(prev_edges, prev_loops, brep_files, data_produced, brep_path):
    """
    Efficiently processes the last BREP file by leveraging previous edges and loops.
    Finds new unique edges and aggregates loops involving at least one new edge.

    Parameters:
    - prev_edges: Tensor of previously processed edge features.
    - prev_loops: List of loops formed by the previous edges.
    - brep_files: List of BREP file names.
    - data_produced: Placeholder for additional data, currently unused.
    - brep_path: Path to the directory containing the BREP files.

    Returns:
    - final_brep_edges: Tensor of updated edge features including the last file.
    - final_loops: Updated loops including new loops formed with the last file.
    """

    # Step 1: Read features from the last file
    last_file = brep_files[-1]
    brep_file_path = os.path.join(brep_path, last_file)
    edge_features_list, cylinder_features = Preprocessing.SBGCN.brep_read.create_graph_from_step_file(brep_file_path)

    # Convert new features to a Tensor
    straight_edges = torch.tensor(edge_features_list, dtype=prev_edges.dtype, device=prev_edges.device)
    new_edges = torch.tensor(edge_features_list + cylinder_features, dtype=prev_edges.dtype, device=prev_edges.device)

    # Step 2: Combine previous and new features
    final_brep_edges = torch.cat((prev_edges, new_edges), dim=0)

    # Convert Tensor to list for padding
    final_brep_edges_list = final_brep_edges.tolist()

    # Step 3: Pad the combined features
    output_brep_edges = Preprocessing.proc_CAD.helper.pad_brep_features(final_brep_edges_list)

    # Step 4: Find unique new edges
    unique_new_edges = Preprocessing.generate_dataset_baseline.find_new_features(prev_edges.tolist(), new_edges.tolist())

    # Step 5: Find loops with the new face_aggregate() function
    new_loops = Preprocessing.proc_CAD.helper.face_aggregate_networkx_accumulate(
        np.array(unique_new_edges), np.array(prev_edges.tolist())
    )

    # Ensure both are lists before concatenation
    new_loops = list(new_loops)  # Convert to list if it is a tuple
    circle_loops = Preprocessing.proc_CAD.helper.face_aggregate_circle_brep(output_brep_edges)

    # Concatenate the lists
    new_loops += list(circle_loops)  # Ensure circle_loops is also a list

    return final_brep_edges, prev_loops + new_loops, straight_edges


# --------------------- Chamfer Distance --------------------- #
def chamfer_distance_brep(gt_brep_edges, output_brep_edges, threshold=0.05):
    """
    Calculates the Chamfer distance between ground truth (GT) BREP edges and output BREP edges,
    considering edge containment.

    Parameters:
    - gt_brep_edges (torch.Tensor): A tensor of shape (num_gt_edges, 6), where each row represents an edge
      with two 3D points (start and end).
    - output_brep_edges (torch.Tensor): A tensor of shape (num_output_edges, 6), where each row represents
      an edge with two 3D points (start and end).
    - threshold (float): Distance threshold to check if the output edges are within acceptable limits.

    Returns:
    - on_right_track (bool): True if all distances are below the threshold, False otherwise.
    - dists (list): A list of minimum distances for each output edge.
    - high_dist_indices (list): A list of indices where dists[i] > threshold.
    """
    def is_point_on_segment(point, segment_start, segment_end, tol=1e-6):
        """Check if a point lies on the segment defined by segment_start and segment_end."""
        segment_vec = segment_end - segment_start
        point_vec = point - segment_start
        cross_product = torch.cross(segment_vec, point_vec)
        # Check collinearity
        if torch.norm(cross_product) > tol:
            return False
        # Check bounds
        dot_product = torch.dot(segment_vec, point_vec)
        if dot_product < 0 or dot_product > torch.dot(segment_vec, segment_vec):
            return False
        return True

    # Ensure inputs are tensors
    if not isinstance(gt_brep_edges, torch.Tensor):
        gt_brep_edges = torch.tensor(gt_brep_edges, dtype=torch.float32)
    if not isinstance(output_brep_edges, torch.Tensor):
        output_brep_edges = torch.tensor(output_brep_edges, dtype=torch.float32)

    dists = []
    high_dist_indices = []

    for output_idx, output_edge in enumerate(output_brep_edges):
        output_start, output_end = output_edge[:3], output_edge[3:6]
        min_distance = float('inf')

        for gt_edge in gt_brep_edges:
            gt_start, gt_end = gt_edge[:3], gt_edge[3:6]

            # Check if the output edge is contained within the gt edge
            if (is_point_on_segment(output_start, gt_start, gt_end) and
                    is_point_on_segment(output_end, gt_start, gt_end)):
                min_distance = 0
                break

            # Calculate distances between corresponding vertices
            dist_start = torch.dist(output_start, gt_start, p=2)
            dist_end = torch.dist(output_end, gt_end, p=2)

            # Calculate distances for reversed vertices (to account for edge reversibility)
            dist_start_reversed = torch.dist(output_start, gt_end, p=2)
            dist_end_reversed = torch.dist(output_end, gt_start, p=2)

            # Find the minimum distance for this pair of edges
            dist = min(dist_start + dist_end, dist_start_reversed + dist_end_reversed)

            # Update the minimum distance for this output edge
            min_distance = min(min_distance, dist)

        dists.append(min_distance)

        # Check if this distance exceeds the threshold
        if min_distance > threshold:
            high_dist_indices.append(output_idx)

    # Determine if all distances are below the threshold
    on_right_track = all(dist < threshold for dist in dists)

    # Additional visualization if there are high-distance indices
    # if len(high_dist_indices) > 0:
    #     combined_brep_edges = torch.cat((output_brep_edges, gt_brep_edges), dim=0)
        # Encoders.helper.vis_brep(gt_brep_edges)
        # Encoders.helper.vis_brep_with_indices(combined_brep_edges, high_dist_indices)
        # print("High distance values:", [dists[i] for i in high_dist_indices])

    return on_right_track, high_dist_indices
