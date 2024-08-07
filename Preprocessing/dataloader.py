from torch.utils.data import Dataset
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm 
import pickle

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
        operations_order_matrix = graph_data['operations_order_matrix']
        gnn_strokeCloud_edges = graph_data['gnn_strokeCloud_edges']
        face_to_stroke = graph_data['face_to_stroke']
        stroke_cloud_coplanar = graph_data['stroke_cloud_coplanar']


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
            brep_to_stroke = torch.zeros(0, dtype=torch.float32)
            edge_features = torch.zeros(0, dtype=torch.float32)
            gnn_brep_edges = torch.zeros((2, 0), dtype=torch.float32)
            brep_stroke_connection = torch.zeros((2, 0), dtype=torch.float32)
            brep_coplanar = torch.zeros((2, 0), dtype=torch.float32)

        else:
            embedding_path = os.path.join(self.data_path, data_dir, 'brep_embedding', f'brep_info_{index}.pkl')
            with open(embedding_path, 'rb') as f:
                embedding_data = pickle.load(f)
            
            brep_to_stroke = embedding_data['brep_to_stroke']
            edge_features = embedding_data['edge_features']
            gnn_brep_edges = embedding_data['gnn_brep_edges']
            brep_stroke_connection = embedding_data['brep_stroke_connection']
            brep_coplanar = embedding_data['brep_coplanar']


        return program, node_features, operations_order_matrix, gnn_strokeCloud_edges, face_to_stroke, stroke_cloud_coplanar, brep_to_stroke, edge_features, gnn_brep_edges, brep_stroke_connection, brep_coplanar

    




def Create_DataLoader_example():
    dataset = Program_Graph_Dataset()

    # Create a DataLoader
    data_loader = DataLoader(dataset, batch_size=1, shuffle=True)
    for batch in tqdm(data_loader):
        node_features, operations_matrix, intersection_matrix, program, face_embeddings, edge_embeddings, vertex_embeddings = batch


def process_batch(batch):
    program, node_features, operations_order_matrix, gnn_strokeCloud_edges, face_to_stroke, stroke_cloud_coplanar, brep_to_stroke, edge_features, gnn_brep_edges, brep_stroke_connection, brep_coplanar = batch

    node_features = node_features.to(torch.float32).to(device).squeeze(0)
    edge_features = torch.tensor(edge_features)
    edge_features = edge_features.to(torch.float32).to(device)
    gnn_strokeCloud_edges = gnn_strokeCloud_edges.to(torch.float32).to(device).squeeze(0)
    gnn_brep_edges = gnn_brep_edges.to(torch.float32).to(device).squeeze(0)
    brep_stroke_connection = brep_stroke_connection.to(torch.float32).to(device).squeeze(0)
    stroke_cloud_coplanar = stroke_cloud_coplanar.to(torch.float32).to(device).squeeze(0)
    brep_coplanar = brep_coplanar.to(torch.float32).to(device).squeeze(0)

    return program, node_features, operations_order_matrix, gnn_strokeCloud_edges, face_to_stroke, stroke_cloud_coplanar, brep_to_stroke, edge_features, gnn_brep_edges, brep_stroke_connection, brep_coplanar
