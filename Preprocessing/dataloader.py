from torch.utils.data import Dataset
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm 
import pickle
import numpy as np
from torch_geometric.data import Batch as PyGBatch

import Preprocessing.proc_CAD.helper
import Preprocessing.gnn_graph

import Models.loop_embeddings

class Program_Graph_Dataset(Dataset):
    def __init__(self, dataset, return_data_path=False):
        self.data_path = os.path.join(os.getcwd(), dataset)
        self.data_dirs = [d for d in os.listdir(self.data_path) if os.path.isdir(os.path.join(self.data_path, d))]
        self.index_mapping = self._create_index_mapping()

        self.return_data_path = return_data_path

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
        data_path = os.path.join(self.data_path, data_dir)

        index = shape_file_path_relative.split('_')[-1].split('.')[0]

        # 1) Load Program
        program_file_path = os.path.join(data_path, 'Program.json')
        program_whole = Preprocessing.proc_CAD.helper.program_to_string(program_file_path)
        program = program_whole[:int(index)+2]

        # 2) Load shape data
        shape_file_path = os.path.join(self.data_path, data_dir, 'shape_info', shape_file_path_relative)
        with open(shape_file_path, 'rb') as f:
            shape_data = pickle.load(f)
        
        stroke_cloud_loops = [list(fset) for fset in shape_data['stroke_cloud_loops']]

        stroke_node_features = shape_data['stroke_node_features']
        strokes_perpendicular = shape_data['strokes_perpendicular']
        output_brep_edges = torch.tensor(shape_data['output_brep_edges'], dtype=torch.float32)
        stroke_operations_order_matrix = torch.tensor(shape_data['stroke_operations_order_matrix'], dtype=torch.float32)

        # Convert remaining numpy arrays to tensors
        
        loop_neighboring_vertical = torch.tensor(shape_data['loop_neighboring_vertical'], dtype=torch.long)
        loop_neighboring_horizontal = torch.tensor(shape_data['loop_neighboring_horizontal'], dtype=torch.long)
        loop_neighboring_contained = torch.tensor(shape_data['loop_neighboring_contained'], dtype=torch.long)

        stroke_to_loop = torch.tensor(shape_data['stroke_to_loop'], dtype=torch.long)
        stroke_to_edge = torch.tensor(shape_data['stroke_to_edge'], dtype=torch.long)

        if self.return_data_path:
            return program, stroke_node_features, data_path

        return program, program_whole, stroke_cloud_loops, stroke_node_features, strokes_perpendicular, output_brep_edges, stroke_operations_order_matrix, loop_neighboring_vertical, loop_neighboring_horizontal,loop_neighboring_contained, stroke_to_loop, stroke_to_edge







def pad_masks(mask, target_size=(400, 1)):
    num_loops = mask.shape[0]
    if num_loops < target_size[0]:
        pad_size = target_size[0] - num_loops
        padded_mask = torch.nn.functional.pad(mask, (0, 0, 0, pad_size), value=-1)
    else:
        padded_mask = mask
    return padded_mask
