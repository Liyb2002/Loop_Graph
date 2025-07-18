import os
import json
import torch
from torch.utils.data import Dataset
import shutil
import re
import numpy as np
import pickle

import Preprocessing.cad2sketch_stroke_features
import Preprocessing.proc_CAD.perturbation_helper

from tqdm import tqdm
from pathlib import Path

class perturbation_dataset_loader(Dataset):
    def __init__(self, target):
        """
        Initializes the dataset generator by setting paths and loading the dataset.
        """

        self.data_path = os.path.join(os.getcwd(), 'dataset', target)

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
        folders_sorted = sorted(folders, key=lambda x: int(os.path.basename(x)))

        if not folders:
            print("No folders found in the dataset directory.")
            return

        total_folders = 0
        target_index = 0
        for folder in folders_sorted:
            folder_index = int(os.path.basename(folder))
            if folder_index <= target_index:
                continue
            
            total_folders += 1
            success_process = self.process_subfolder(os.path.join(self.data_path, folder))

            if not success_process:
                print("remove folder:", folder)
                shutil.rmtree(os.path.join(self.data_path, folder))


    def process_subfolder(self, subfolder_path):
        """
        Processes an individual subfolder by reading JSON files and extracting relevant data.
        """

        print("Perturbing Stroke Cloud:", subfolder_path)
        
        final_edges_file_path = os.path.join(subfolder_path, 'final_edges.json')
        all_edges_file_path = os.path.join(subfolder_path, 'unique_edges.json')
        strokes_dict_path = os.path.join(subfolder_path, 'strokes_dict.json')
        program_path = os.path.join(subfolder_path, 'program.json')

        # Check if required JSON files exist, printing which one is missing
        missing_files = []
        
        if not os.path.exists(final_edges_file_path):
            missing_files.append("final_edges.json")
        if not os.path.exists(all_edges_file_path):
            missing_files.append("unique_edges.json")
        if not os.path.exists(strokes_dict_path):
            missing_files.append("strokes_dict.json")
        if not os.path.exists(program_path):
            missing_files.append("program_path.json")

        if missing_files:
            print(f"Skipping {subfolder_path}: Missing files: {', '.join(missing_files)}")
            return None, None, None

        # Do some vis
        # Load and visualize only feature lines version
        final_edges_data = self.read_json(final_edges_file_path)
        feature_lines = Preprocessing.cad2sketch_stroke_features.extract_feature_lines(final_edges_data)
        line_types = Preprocessing.cad2sketch_stroke_features.extract_line_types(final_edges_data)
        # Preprocessing.cad2sketch_stroke_features.vis_feature_lines(feature_lines)


        # Load and visualize only final edges (feature + construction lines)
        all_lines = Preprocessing.cad2sketch_stroke_features.extract_all_lines(final_edges_data)
        # Preprocessing.cad2sketch_stroke_features.vis_feature_lines(all_lines)

        # Load and visualize only construction lines (construction lines)
        # construction_lines = Preprocessing.cad2sketch_stroke_features.extract_only_construction_lines(final_edges_data)
        # Preprocessing.cad2sketch_stroke_features.vis_feature_lines(construction_lines)


        stroke_node_features, _= Preprocessing.cad2sketch_stroke_features.build_final_edges_json(final_edges_data)
        all_lines, stroke_node_features = Preprocessing.proc_CAD.perturbation_helper.remove_contained_lines(all_lines, stroke_node_features)
        all_lines, stroke_node_features = Preprocessing.proc_CAD.perturbation_helper.duplicate_lines(all_lines, stroke_node_features)

        all_lines = Preprocessing.proc_CAD.perturbation_helper.compute_opacity(all_lines)

        perturbed_all_lines = Preprocessing.proc_CAD.perturbation_helper.do_perturb(all_lines, stroke_node_features)
        # Preprocessing.cad2sketch_stroke_features.vis_feature_lines(all_lines)
        Preprocessing.cad2sketch_stroke_features.vis_feature_lines(perturbed_all_lines)

        perturbed_output_path = os.path.join(subfolder_path, 'perturbed_all_lines.json')

        # Save to JSON file
        with open(perturbed_output_path, 'w') as f:
            json.dump(perturbed_all_lines, f, indent=4)


        return True




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



# target = 'cad2sketch_annotated'
# d_generator = Preprocessing.perturb_stroke_cloud.perturbation_dataset_loader(target)
