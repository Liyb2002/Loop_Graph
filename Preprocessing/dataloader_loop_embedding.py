from torch.utils.data import Dataset
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm 
import pickle
import numpy as np

from Preprocessing.config import device
import Preprocessing.proc_CAD.helper
import Preprocessing.SBGCN.run_SBGCN

class Program_Graph_Dataset(Dataset):
    def __init__(self, dataset):
        self.data_path = os.path.join(os.getcwd(), dataset)
        self.data_dirs = [d for d in os.listdir(self.data_path) if os.path.isdir(os.path.join(self.data_path, d))]
        self.index_mapping = self._create_index_mapping()

        print(f"Number of data directories: {len(self.data_dirs)}")
        print(f"Total number of brep_i.step files: {len(self.index_mapping)}")

    def _create_index_mapping(self):
        index_mapping = []
        for data_dir in self.data_dirs:
            shape_info_path = os.path.join(self.data_path, data_dir, 'shape_info')
            if os.path.exists(shape_info_path):
                shape_files = sorted([f for f in os.listdir(shape_info_path) if f.endswith('.pkl')])
                for shape_file in shape_files:
                    index_mapping.append((data_dir, shape_file))
        return index_mapping

    def __len__(self):
        return len(self.index_mapping)


    def __getitem__(self, idx):
        data_dir, shape_file_path_relative = self.index_mapping[idx]

        # 1) Load shape data
        shape_file_path = os.path.join(self.data_path, data_dir, 'shape_info', shape_file_path_relative)
        with open(shape_file_path, 'rb') as f:
            shape_data = pickle.load(f)
        
        stroke_cloud_loops = [list(fset) for fset in shape_data['stroke_cloud_loops']]
        stroke_node_features = shape_data['stroke_node_features']

        loop_neighboring_vertical = shape_data['loop_neighboring_vertical']
        loop_neighboring_horizontal = shape_data['loop_neighboring_horizontal']
        loop_neighboring_combined = np.logical_or(loop_neighboring_vertical, loop_neighboring_horizontal).astype(int)

        return stroke_cloud_loops, stroke_node_features, loop_neighboring_combined
        
    


def combine_matrix(loop_neighboring_vertical, loop_neighboring_horizontal):
    
    return np.logical_or(loop_neighboring_vertical, loop_neighboring_horizontal).astype(int)

