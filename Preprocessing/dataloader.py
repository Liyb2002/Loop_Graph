from torch.utils.data import Dataset
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm 
import pickle

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
            canvas_dir_path = os.path.join(self.data_path, data_dir, 'canvas')
            if os.path.exists(canvas_dir_path):
                brep_files = sorted([f for f in os.listdir(canvas_dir_path) if f.startswith('brep_') and f.endswith('.step')])
                for brep_file_path in brep_files:
                    index_mapping.append((data_dir, brep_file_path))
                index_mapping.append((data_dir, '-1'))
        return index_mapping

    def __len__(self):
        return len(self.index_mapping)


    def __getitem__(self, idx):
        data_dir, brep_file_path = self.index_mapping[idx]
        data_path = os.path.join(self.data_path, data_dir)

        if brep_file_path == '-1':
            index = brep_file_path
        else:
            index = brep_file_path.split('_')[1].split('.')[0]

        # 1) Load graph
        graph_path = os.path.join(data_path, 'stroke_cloud_graph.pkl')
        with open(graph_path, 'rb') as f:
            graph_data = pickle.load(f)
        
        # Four matrices to build the graph
        node_features = graph_data['node_features']
        operations_matrix = graph_data['operations_matrix']
        intersection_matrix = graph_data['intersection_matrix']
        operations_order_matrix = graph_data['operations_order_matrix']
        face_to_stroke = graph_data['face_aggregate']


        # 2) Load Program
        # Program Len = brep Len + 1
        # index = -1 -> brep: empty file; program[0]
        # index = 0 -> brep: brep_0.step; program[0, 1]
        # index = 1 -> brep: brep_1.step; program[0, 1, 2]
        # final case: index = 5 -> brep : brep_{5}.step; program[0,1,2,3,4,5,6]

        # program[0] makes brep_0.step, given brep_0.step, we want to predict program[1]
        program_file_path = os.path.join(data_path, 'Program.json')
        program = Preprocessing.proc_CAD.helper.program_to_string(program_file_path)
        program = program[:int(index)+2]
        program = Preprocessing.proc_CAD.helper.program_to_tensor(program)

        # 2.5) If next program operation is sketch, need to know which face we are using
        face_boundary_points = Preprocessing.proc_CAD.helper.sketch_face_selection(program_file_path)

        # 3) Load Brep embedding
        if int(index) == -1:
            face_feature_gnn_list = torch.empty(0, dtype=torch.float32)
            face_features = torch.empty(0, dtype=torch.float32)
            edge_features = torch.empty(0, dtype=torch.float32)
            vertex_features = torch.empty(0, dtype=torch.float32)
            edge_index_face_edge_list = torch.empty((2, 0), dtype=torch.long)
            edge_index_edge_vertex_list = torch.empty((2, 0), dtype=torch.long)
            edge_index_face_face_list = torch.empty((2, 0), dtype=torch.long)
            index_id = torch.empty(0, dtype=torch.long)

        else:
            embedding_path = os.path.join(self.data_path, data_dir, 'brep_embedding', f'brep_info_{index}.pkl')
            with open(embedding_path, 'rb') as f:
                embedding_data = pickle.load(f)
            
            face_feature_gnn_list = embedding_data['face_feature_gnn_list']
            face_features = embedding_data['face_features']
            edge_features = embedding_data['edge_features']
            vertex_features = embedding_data['vertex_features']
            edge_index_face_edge_list = embedding_data['edge_index_face_edge_list']
            edge_index_edge_vertex_list = embedding_data['edge_index_edge_vertex_list']
            edge_index_face_face_list = embedding_data['edge_index_face_face_list']
            index_id = embedding_data['index_id']

            
        return node_features, operations_matrix, intersection_matrix, operations_order_matrix, face_to_stroke, program, face_boundary_points, face_feature_gnn_list, face_features, edge_features, vertex_features, edge_index_face_edge_list, edge_index_edge_vertex_list, edge_index_face_face_list, index_id

    




def Create_DataLoader_example():
    dataset = Program_Graph_Dataset()

    # Create a DataLoader
    data_loader = DataLoader(dataset, batch_size=1, shuffle=True)
    for batch in tqdm(data_loader):
        node_features, operations_matrix, intersection_matrix, program, face_embeddings, edge_embeddings, vertex_embeddings = batch
