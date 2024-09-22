from torch.utils.data import Dataset
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm 
import pickle
import numpy as np
from torch_geometric.data import Batch as PyGBatch

from Preprocessing.config import device
import Preprocessing.proc_CAD.helper
import Preprocessing.SBGCN.run_SBGCN
import Preprocessing.gnn_graph

import Models.loop_embeddings

class Program_Graph_Dataset(Dataset):
    def __init__(self, dataset):
        self.data_path = os.path.join(os.getcwd(), dataset)
        self.data_dirs = [d for d in os.listdir(self.data_path) if os.path.isdir(os.path.join(self.data_path, d))]
        self.index_mapping = self._create_index_mapping()

        print(f"Number of data directories: {len(self.data_dirs)}")
        print(f"Total number of brep_i.step files: {len(self.index_mapping)}")

        self.load_loop_embed_model()

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


    def load_loop_embed_model(self):
        current_dir = os.getcwd()
        loop_embedding_dir = os.path.join(current_dir, 'checkpoints', 'loop_embedding_model')

        self.loop_embed_model = Models.loop_embeddings.LoopEmbeddingNetwork()
        self.loop_embed_model.load_state_dict(torch.load(os.path.join(loop_embedding_dir, 'loop_embed_model.pth')))
        self.loop_embed_model.to(device)
        self.loop_embed_model.eval()


    def __getitem__(self, idx):
        data_dir, shape_file_path_relative = self.index_mapping[idx]
        data_path = os.path.join(self.data_path, data_dir)

        index = shape_file_path_relative.split('_')[-1].split('.')[0]

        # 1) Load Program
        program_file_path = os.path.join(data_path, 'Program.json')
        program_whole = Preprocessing.proc_CAD.helper.program_to_string(program_file_path)
        program = self.get_program(program_whole, idx)

        # 2) Load shape data
        shape_file_path = os.path.join(self.data_path, data_dir, 'shape_info', shape_file_path_relative)
        with open(shape_file_path, 'rb') as f:
            shape_data = pickle.load(f)
        
        stroke_cloud_loops = [list(fset) for fset in shape_data['stroke_cloud_loops']]
        stroke_node_features = shape_data['stroke_node_features']

        # Convert remaining numpy arrays to tensors
        
        loop_neighboring_vertical = torch.tensor(shape_data['loop_neighboring_vertical'], dtype=torch.long, device=device)
        loop_neighboring_horizontal = torch.tensor(shape_data['loop_neighboring_horizontal'], dtype=torch.long, device=device)
        loop_neighboring_contained = torch.tensor(shape_data['loop_neighboring_contained'], dtype=torch.long, device=device)

        stroke_to_brep = torch.tensor(shape_data['stroke_to_brep'], dtype=torch.long, device=device)
        final_brep_edges = torch.tensor(shape_data['final_brep_edges'], dtype=torch.float32, device=device)

        # Load stroke_operations_order_matrix and convert to tensor
        stroke_operations_order_matrix = torch.tensor(shape_data['stroke_operations_order_matrix'], dtype=torch.float32)

        return stroke_cloud_loops, stroke_node_features,loop_neighboring_vertical, loop_neighboring_horizontal,loop_neighboring_contained, stroke_to_brep, stroke_operations_order_matrix, final_brep_edges




    def get_program(self, program, idx):
        sketch_count = 0
        result = []
        
        for i, action in enumerate(program):
            result.append(action)
            
            # Increment the 'sketch' count if the current action is 'sketch'
            if action == 'sketch':
                sketch_count += 1
            
            if sketch_count == idx + 2:
                break

        return result


