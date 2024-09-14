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
        
        stroke_cloud_loops = [list(fset) for fset in shape_data['stroke_cloud_loops']]
        stroke_node_features = shape_data['stroke_node_features']

        # Prepare loop features
        loop_features = []
        for indices in stroke_cloud_loops:
            strokes = stroke_node_features[indices]  # Extract strokes for current loop

            # If there are only 3 strokes, pad to have 4 strokes
            if strokes.shape[0] == 3:
                padding = np.zeros((1, strokes.shape[1]))  # Create a zero padding
                strokes = np.vstack([strokes, padding])  # Pad to shape (4, 6)

            # Flatten the strokes to create a single feature vector for the loop
            loop_feature = strokes.flatten()  # Shape: (1, 24)
            loop_features.append(loop_feature)

        # Convert to tensor
        loop_features = torch.tensor(loop_features, dtype=torch.float32)  # Shape: (len(stroke_cloud_loops), 24)


        loop_neighboring_vertical = shape_data['loop_neighboring_vertical']
        loop_neighboring_horizontal = shape_data['loop_neighboring_horizontal']
        loop_neighboring_combined = np.logical_or(loop_neighboring_vertical, loop_neighboring_horizontal).astype(int)
        loop_neighboring_combined = torch.tensor(loop_neighboring_combined, dtype=torch.float32)

        return loop_features, loop_neighboring_combined



def custom_collate_fn(batch):
    # Separate loop_features and loop_neighboring_combined from the batch
    loop_features_list = [item[0] for item in batch]  # List of (num_loops, 24)
    loop_neighboring_combined_list = [item[1] for item in batch]  # List of (num_loops, num_loops)

    # Find the maximum number of loops in this batch
    max_num_loops = max(f.shape[0] for f in loop_features_list)

    # Pad loop_features to the maximum number of loops in the batch
    padded_loop_features = torch.stack([
        torch.cat([f.to(device), torch.zeros(max_num_loops - f.shape[0], f.shape[1], device=device)], dim=0)
        for f in loop_features_list
    ])  # Shape: (batch_size, max_num_loops, 24)

    # Pad loop_neighboring_combined to match the maximum number of loops
    padded_loop_neighboring_combined = torch.stack([
        torch.nn.functional.pad(m.to(device), (0, max_num_loops - m.shape[1], 0, max_num_loops - m.shape[0]), value=0)
        for m in loop_neighboring_combined_list
    ])  # Shape: (batch_size, max_num_loops, max_num_loops)

    # Create masks to indicate valid entries (1 for valid, 0 for padding)
    mask_loop_features = torch.tensor([
        [1] * f.shape[0] + [0] * (max_num_loops - f.shape[0])
        for f in loop_features_list
    ], dtype=torch.float32, device=device)  # Shape: (batch_size, max_num_loops)

    return padded_loop_features, padded_loop_neighboring_combined, mask_loop_features
