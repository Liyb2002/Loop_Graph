import os
import json
import torch
from torch.utils.data import Dataset
import shutil
import re
import numpy as np
import pickle

import Preprocessing.cad2sketch_stroke_features


import Preprocessing.proc_CAD.proc_gen
import Preprocessing.proc_CAD.CAD_to_stroke_cloud
import Preprocessing.proc_CAD.render_images
import Preprocessing.proc_CAD.Program_to_STL
import Preprocessing.proc_CAD.helper
import Preprocessing.proc_CAD.render_images
import Preprocessing.proc_CAD.draw_all_lines_baseline

import Preprocessing.gnn_graph
import Preprocessing.SBGCN.brep_read

from tqdm import tqdm
from pathlib import Path

class cad2sketch_dataset_loader(Dataset):
    def __init__(self):
        """
        Initializes the dataset generator by setting paths and loading the dataset.
        """

        self.data_path = os.path.join(os.getcwd(), 'dataset', 'selected_dataset')

        self.subfolder_paths = []

        self.load_dataset()


    def load_dataset(self):
        """
        Loads the dataset by iterating over all subfolders and storing their paths.
        """
        if not os.path.exists(self.data_path):
            print(f"Dataset path '{self.data_path}' not found.")
            return

        folders = [folder for folder in os.listdir(self.data_path) if os.path.isdir(os.path.join(self.data_path, folder))]

        if not folders:
            print("No folders found in the dataset directory.")
            return

        # folder = 1600
        for folder in folders:
            folder_path = os.path.join(self.data_path, folder)
            subfolders = [sf for sf in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, sf))]

            if not subfolders:
                print(f"No subfolders found in '{folder}'. Skipping...")
                continue

            # subfolder = 49.39_144.98
            for subfolder in subfolders:
                subfolder_path = os.path.join(folder_path, subfolder)
                self.subfolder_paths.append(subfolder_path)  # Store paths instead of processing
                break
            

        for subfolder_path in tqdm(self.subfolder_paths, desc=f"Cleaning Data",):
            self.process_subfolder( subfolder_path)

    # IDEA:
    # We are in /selected_dataset/1600
    # Henro's code will create a /canvas folder that put all the .step files and the rotation matrix
    # process_subfolder() will give a /shape_info folder that stores all the .pkl files
    def process_subfolder(self, subfolder_path):
        """
        Processes an individual subfolder by reading JSON files and extracting relevant data.
        """
        final_edges_file_path = os.path.join(subfolder_path, 'final_edges.json')
        all_edges_file_path = os.path.join(subfolder_path, 'unique_edges.json')
        strokes_dict_path = os.path.join(subfolder_path, 'strokes_dict.json')

        # Check if required JSON files exist, printing which one is missing
        missing_files = []
        
        if not os.path.exists(final_edges_file_path):
            missing_files.append("final_edges.json")
        if not os.path.exists(all_edges_file_path):
            missing_files.append("unique_edges.json")
        if not os.path.exists(strokes_dict_path):
            missing_files.append("strokes_dict.json")

        if missing_files:
            # print(f"Skipping {subfolder_path}: Missing files: {', '.join(missing_files)}")
            return None, None, None

        # Do some vis
        # Load and visualize only feature lines version
        final_edges_data = self.read_json(final_edges_file_path)
        # feature_lines = Preprocessing.cad2sketch_stroke_features.extract_feature_lines(final_edges_data)
        # Preprocessing.cad2sketch_stroke_features.vis_feature_lines(feature_lines)


        # Load and visualize only final edges (feature + construction lines)
        all_lines = Preprocessing.cad2sketch_stroke_features.extract_all_lines(final_edges_data)
        # Preprocessing.cad2sketch_stroke_features.vis_feature_lines(all_lines)


        # Load and visualize only construction lines (construction lines)
        # construction_lines = Preprocessing.cad2sketch_stroke_features.extract_only_construction_lines(final_edges_data)
        # Preprocessing.cad2sketch_stroke_features.vis_feature_lines(construction_lines)


        # ------------------------------------------------------------ #
        # Now start information processing
        stroke_node_features, is_feature_line_matrix= Preprocessing.cad2sketch_stroke_features.build_final_edges_json(final_edges_data)
        stroke_node_features, added_feature_lines= Preprocessing.cad2sketch_stroke_features.split_and_merge_stroke_cloud(stroke_node_features, is_feature_line_matrix)
        # Preprocessing.cad2sketch_stroke_features.vis_stroke_node_features_and_highlights(stroke_node_features, added_feature_lines)

        stroke_operations_order_matrix = None


        connected_stroke_nodes = Preprocessing.proc_CAD.helper.connected_strokes(stroke_node_features)
        strokes_perpendicular, strokes_non_perpendicular =  Preprocessing.proc_CAD.helper.stroke_relations(stroke_node_features, connected_stroke_nodes)


        # 2) Get the loops
        stroke_cloud_loops = Preprocessing.proc_CAD.helper.face_aggregate_networkx(stroke_node_features) + Preprocessing.proc_CAD.helper.face_aggregate_circle(stroke_node_features)
        stroke_cloud_loops = Preprocessing.proc_CAD.helper.reorder_loops(stroke_cloud_loops)
        stroke_cloud_loops = [list(loop) for loop in stroke_cloud_loops]
        # Preprocessing.cad2sketch_stroke_features.vis_feature_lines_loop_all(all_lines, stroke_cloud_loops)


        # 3) Compute Loop Neighboring Information
        loop_neighboring_all = Preprocessing.proc_CAD.helper.loop_neighboring_simple(stroke_cloud_loops)
        loop_neighboring_vertical = Preprocessing.proc_CAD.helper.loop_neighboring_complex(stroke_cloud_loops, stroke_node_features, loop_neighboring_all)
        loop_neighboring_horizontal = Preprocessing.proc_CAD.helper.coplanr_neighorbing_loop(loop_neighboring_all, loop_neighboring_vertical)
        loop_neighboring_contained = Preprocessing.proc_CAD.helper.loop_contained(stroke_cloud_loops, stroke_node_features)


        # get the brep generation process
        parent_folder = os.path.dirname(subfolder_path)
        brep_folder_path = os.path.join(parent_folder, 'canvas')
        if os.path.exists(brep_folder_path) and os.path.isdir(brep_folder_path):
            step_files = [f for f in os.listdir(brep_folder_path) if f.endswith('.step')]
            step_files.sort(key=lambda x: int(re.search(r'step_(\d+)\.step', x).group(1)) if re.search(r'step_(\d+)\.step', x) else float('inf'))


        # now, process the brep files
    
        final_brep_edges = []
        final_cylinder_features = []
        new_features = []

        matrix_path = os.path.join(parent_folder, 'canvas', 'matrix.json')
        with open(matrix_path, 'r') as f:
            rotation_matrix = json.load(f)


        stroke_operations_order_matrix = np.zeros((stroke_node_features.shape[0], len(step_files)))
        data_directory = os.path.join(parent_folder, 'shape_info')
        file_count = 0
        for idx, step_file in enumerate(step_files):
            edge_features_list, cylinder_features = Preprocessing.SBGCN.brep_read.create_graph_from_step_file(os.path.join(brep_folder_path, step_file))
            edge_features_list, cylinder_features= Preprocessing.cad2sketch_stroke_features.rotate_matrix(edge_features_list, cylinder_features, rotation_matrix)
            edge_features_list += Preprocessing.cad2sketch_stroke_features.split_and_stick(edge_features_list)

            if len(final_brep_edges) == 0:
                new_features = edge_features_list
                new_features_cylinder = cylinder_features

                final_brep_edges = edge_features_list
                final_cylinder_features = cylinder_features
            else:
                # We already have brep
                new_features = Preprocessing.cad2sketch_stroke_features.find_new_features_simple(final_brep_edges, edge_features_list) 
                new_features_cylinder = Preprocessing.cad2sketch_stroke_features.find_new_features_simple(final_cylinder_features, cylinder_features)

                final_brep_edges += new_features
                final_cylinder_features += new_features_cylinder
        
            output_brep_edges = Preprocessing.proc_CAD.helper.pad_brep_features(final_brep_edges + final_cylinder_features)
            Preprocessing.cad2sketch_stroke_features.vis_brep(output_brep_edges)

            # 5) Stroke_Cloud - Brep Connection
            stroke_to_edge_lines = Preprocessing.proc_CAD.helper.stroke_to_edge(stroke_node_features, output_brep_edges)
            stroke_to_edge_circle = Preprocessing.proc_CAD.helper.stroke_to_edge_circle_full(stroke_node_features, output_brep_edges)
            stroke_to_edge = Preprocessing.proc_CAD.helper.union_matrices(stroke_to_edge_lines, stroke_to_edge_circle)
            # Preprocessing.cad2sketch_stroke_features.vis_feature_lines_selected(all_lines, stroke_to_edge)

            stroke_to_loop = Preprocessing.cad2sketch_stroke_features.from_stroke_to_edge(stroke_to_edge, stroke_cloud_loops)
            # Preprocessing.cad2sketch_stroke_features.vis_feature_lines_loop_ver(all_lines, stroke_to_loop, stroke_cloud_loops)

            # 6) We need to build the stroke_operations_order_matrix
            new_stroke_to_edge_straight = Preprocessing.proc_CAD.helper.stroke_to_edge(stroke_node_features, new_features)
            new_stroke_to_edge_circle = Preprocessing.proc_CAD.helper.stroke_to_edge_circle(stroke_node_features, new_features_cylinder)
            new_stroke_to_edge_matrix = Preprocessing.proc_CAD.helper.union_matrices(new_stroke_to_edge_straight, new_stroke_to_edge_circle)
            
            stroke_operations_order_matrix[:, idx] = np.array(new_stroke_to_edge_matrix).flatten()

            # Preprocessing.cad2sketch_stroke_features.vis_feature_lines_selected(all_lines, new_stroke_to_edge_matrix)

            # 7) Write the data to file
            os.makedirs(os.path.join(data_directory, 'shape_info'), exist_ok=True)
            output_file_path = os.path.join(data_directory, f'shape_info_{file_count}.pkl')
            with open(output_file_path, 'wb') as f:
                pickle.dump({
                    'stroke_cloud_loops': stroke_cloud_loops, 

                    'stroke_node_features': stroke_node_features,
                    'stroke_type_features': np.matrix([]),
                    'strokes_perpendicular': strokes_perpendicular,
                    'output_brep_edges': output_brep_edges,
                    'stroke_operations_order_matrix': stroke_operations_order_matrix, 

                    'loop_neighboring_vertical': loop_neighboring_vertical,
                    'loop_neighboring_horizontal': loop_neighboring_horizontal,
                    'loop_neighboring_contained': loop_neighboring_contained,

                    'stroke_to_loop': stroke_to_loop,
                    'stroke_to_edge': stroke_to_edge
                }, f)
            
            file_count += 1
  
        return None


    def __getitem__(self, index):
        """
        Loads and processes the next subfolder when requested.
        If a subfolder has missing files, find the next available subfolder.
        Returns a tuple (intersection_matrix, all_edges_matrix, final_edges_matrix).
        """
        pass

    def __len__(self):
        """
        Returns the total number of items in the dataset.
        """
        return len(self.subfolder_paths)

    def read_json(self, file_path):
        """
        Reads a JSON file and returns its contents.
        """
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error reading JSON file {file_path}: {e}")
            return None
