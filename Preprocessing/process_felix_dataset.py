import os
import json
import torch
from torch.utils.data import Dataset
import shutil
import re


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
            

        for subfolder_path in tqdm(self.subfolder_paths, desc=f"Cleaning Data",):
            self.process_subfolder( subfolder_path)


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
        feature_lines = Preprocessing.cad2sketch_stroke_features.extract_feature_lines(final_edges_data)
        # Preprocessing.cad2sketch_stroke_features.vis_feature_lines(feature_lines)


        # Load and visualize only final edges (feature + construction lines)
        all_lines = Preprocessing.cad2sketch_stroke_features.extract_all_lines(final_edges_data)
        # Preprocessing.cad2sketch_stroke_features.vis_feature_lines(all_lines)


        # Load and visualize only construction lines (construction lines)
        construction_lines = Preprocessing.cad2sketch_stroke_features.extract_only_construction_lines(final_edges_data)
        # Preprocessing.cad2sketch_stroke_features.vis_feature_lines(construction_lines)


        # ------------------------------------------------------------ #
        # Now start information processing
        stroke_node_features = Preprocessing.cad2sketch_stroke_features.build_final_edges_json(final_edges_data)


        # get the brep generation process
        parent_folder = os.path.dirname(subfolder_path)
        output_folder_path = os.path.join(parent_folder, 'output', 'canvas')
        if os.path.exists(output_folder_path) and os.path.isdir(output_folder_path):
            step_files = [f for f in os.listdir(output_folder_path) if f.endswith('.step')]
            step_files.sort(key=lambda x: int(re.search(r'step_(\d+)\.step', x).group(1)) if re.search(r'step_(\d+)\.step', x) else float('inf'))


        # now, process the brep files
        for step_file in step_files:
            edge_features_list, cylinder_features = Preprocessing.SBGCN.brep_read.create_graph_from_step_file(os.path.join(output_folder_path, step_file))

            print("edge_features_list", len(edge_features_list))
            print("cylinder_features", len(cylinder_features))
            print("----")

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
