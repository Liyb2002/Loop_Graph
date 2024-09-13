from torch.utils.data import Dataset
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm 
import pickle
import numpy as np
from torch.nn.utils.rnn import pad_sequence

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

        stroke_cloud_loops = [torch.tensor(list(fset), dtype=torch.float32) for fset in shape_data['stroke_cloud_loops']]
        stroke_cloud_loops_padded = pad_sequence(stroke_cloud_loops, batch_first=True)

        # Convert other data to tensors
        stroke_node_features = torch.tensor(shape_data['stroke_node_features'], dtype=torch.float32)
        
        loop_neighboring_vertical = shape_data['loop_neighboring_vertical']
        loop_neighboring_horizontal = shape_data['loop_neighboring_horizontal']
        loop_neighboring_combined = torch.tensor(np.logical_or(loop_neighboring_vertical, loop_neighboring_horizontal).astype(int), dtype=torch.float32)

        return stroke_cloud_loops_padded, stroke_node_features, loop_neighboring_combined
    


def custom_collate_fn(batch):
    stroke_cloud_loops_list, stroke_node_features_list, loop_neighboring_combined_list = [], [], []
    stroke_node_features_lengths = []
    loop_neighboring_combined_shapes = []

    for item in batch:
        stroke_cloud_loops, stroke_node_features, loop_neighboring_combined = item
        stroke_cloud_loops_list.extend(stroke_cloud_loops)  # Flatten list of loops
        stroke_node_features_list.append(stroke_node_features)
        loop_neighboring_combined_list.append(loop_neighboring_combined)
        stroke_node_features_lengths.append(stroke_node_features.shape[0])
        loop_neighboring_combined_shapes.append(loop_neighboring_combined.shape)

    # Find the maximum length in the batch
    max_stroke_node_features_length = max(stroke_node_features_lengths)
    max_loop_neighboring_combined_dim = max(max(shape) for shape in loop_neighboring_combined_shapes)

    # Pad stroke_node_features to the maximum length
    padded_stroke_node_features = torch.stack([
        torch.cat([f, torch.zeros(max_stroke_node_features_length - f.shape[0], f.shape[1])], dim=0)
        for f in stroke_node_features_list
    ])

    # Pad loop_neighboring_combined to the maximum length for both dimensions
    padded_loop_neighboring_combined = torch.stack([
        torch.nn.functional.pad(m, (0, max_loop_neighboring_combined_dim - m.shape[1], 0, max_loop_neighboring_combined_dim - m.shape[0]), value=0)
        for m in loop_neighboring_combined_list
    ])

    # Create masks to indicate the valid lengths (1 for real data, 0 for padding)
    mask_stroke_node_features = torch.tensor([
        [1] * length + [0] * (max_stroke_node_features_length - length)
        for length in stroke_node_features_lengths
    ], dtype=torch.float32)

    mask_loop_neighboring_combined = torch.stack([
        torch.nn.functional.pad(torch.ones(m.shape), (0, max_loop_neighboring_combined_dim - m.shape[1], 0, max_loop_neighboring_combined_dim - m.shape[0]), value=0)
        for m in loop_neighboring_combined_list
    ])

    return stroke_cloud_loops_list, padded_stroke_node_features, padded_loop_neighboring_combined, mask_stroke_node_features, mask_loop_neighboring_combined
