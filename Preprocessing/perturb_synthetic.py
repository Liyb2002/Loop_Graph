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

import Preprocessing.cad2sketch_stroke_features
import Preprocessing.perturb_stroke_cloud_reverse

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
            shape_info_dir = os.path.join(self.data_path, dir, 'shape_info')
            pattern = re.compile(r'shape_info_(\d+)\.pkl')

            max_x = -1
            max_file = None

            for filename in os.listdir(shape_info_dir):
                match = pattern.match(filename)
                if match:
                    x = int(match.group(1))
                    if x > max_x:
                        max_x = x
                        max_file = filename

            
            if max_file is None:
                continue
            
            full_path = os.path.join(shape_info_dir, max_file)
            with open(full_path, 'rb') as f:
                base_shape_data = pickle.load(f)
            
            stroke_node_features = base_shape_data['stroke_node_features']
            stroke_cloud = Preprocessing.perturb_stroke_cloud_reverse.stroke_node_features_to_polyline(stroke_node_features)
            # Preprocessing.cad2sketch_stroke_features.vis_stroke_node_features(stroke_node_features)
            Preprocessing.cad2sketch_stroke_features.vis_feature_lines(stroke_cloud)



    def __getitem__(self, idx):
        return