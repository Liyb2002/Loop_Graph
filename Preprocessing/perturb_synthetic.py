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
    def __init__(self):
        self.data_path = os.path.join(os.getcwd(), 'dataset', 'whole')
        self.data_dirs = [d for d in os.listdir(self.data_path) if os.path.isdir(os.path.join(self.data_path, d))]
        
        print(f"Number of data directories: {len(self.data_dirs)}")

        self.do_perturbation()


    def __len__(self):
        return len(self.index_mapping)


    def do_perturbation(self):
        for dir in self.data_dirs:
            print('dir', dir)


    def __getitem__(self, idx):
        data_dir, shape_file_path_relative = self.index_mapping[idx]
        data_path = os.path.join(self.data_path, data_dir)

        index = shape_file_path_relative.split('_')[-1].split('.')[0]

            

        # 1) Load Program
        program_file_path = os.path.join(data_path, 'Program.json')
        program_whole = Preprocessing.proc_CAD.helper.program_to_string(program_file_path)
        program = program_whole[:int(index)+2]

        # 2) Load basic shape data
        shape_info_dir = os.path.join(self.data_path, data_dir, 'shape_info')
        if not os.path.exists(shape_info_dir):
            return None
        
        shape_files = [f for f in os.listdir(shape_info_dir) if re.match(r'shape_info_(\d+)\.pkl', f)]
        shape_numbers = [int(re.search(r'shape_info_(\d+)\.pkl', f).group(1)) for f in shape_files]

        if len(shape_numbers) == 0:
            return None
        
        base_shape_file_path = os.path.join(self.data_path, data_dir, 'shape_info', f'shape_info_{max(shape_numbers)}.pkl')
        with open(base_shape_file_path, 'rb') as f:
            base_shape_data = pickle.load(f)


        stroke_cloud_loops = [list(fset) for fset in base_shape_data['stroke_cloud_loops']]
        stroke_node_features = base_shape_data['stroke_node_features']

        
        return data_dir, program, program_whole, stroke_cloud_loops, stroke_node_features, strokes_perpendicular, output_brep_edges, stroke_operations_order_matrix, loop_neighboring_vertical, loop_neighboring_horizontal,loop_neighboring_contained, stroke_to_loop, stroke_to_edge





