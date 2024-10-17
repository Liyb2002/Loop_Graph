
from torch.utils.data import Dataset, DataLoader
import os
import pickle
from tqdm import tqdm

import Preprocessing.SBGCN.brep_read
import Preprocessing.proc_CAD.helper

# --------------------- Loader for output --------------------- #
class Evaluation_Dataset(Dataset):
    def __init__(self, dataset):
        self.data_path = os.path.join(os.getcwd(), dataset)
        self.data_dirs = [d for d in os.listdir(self.data_path) if os.path.isdir(os.path.join(self.data_path, d))]

        print(f"Number of data directories: {len(self.data_dirs)}")

    def __len__(self):
        return len(self.data_dirs)

    def __getitem__(self, idx):
        data_dir = self.data_dirs[idx]

        # Load shape_info
        shape_file_path = os.path.join(self.data_path, data_dir, 'shape_info.pkl')
        with open(shape_file_path, 'rb') as f:
            shape_data = pickle.load(f)
        stroke_node_features = shape_data['stroke_node_features']
        

        # Load Brep file
        brep_files = [file_name for file_name in os.listdir(os.path.join(self.data_path, data_dir, 'canvas'))
                if file_name.startswith('brep_') and file_name.endswith('.step')]
        brep_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))


        final_brep_file_path = os.path.join(self.data_path, data_dir, 'canvas', brep_files[-1])
        edge_features_list, cylinder_features = Preprocessing.SBGCN.brep_read.create_graph_from_step_file(final_brep_file_path)
        output_brep_edges = Preprocessing.proc_CAD.helper.pad_brep_features(edge_features_list + cylinder_features)


        return stroke_node_features, output_brep_edges



# --------------------- Main Code --------------------- #

# Set up dataloader
dataset = Evaluation_Dataset('program_output')
data_loader = DataLoader(dataset, batch_size=1, shuffle=True)


for data in tqdm(data_loader, desc="Generating CAD Programs"):
    stroke_node_features, output_brep_edges = data
    print("output_brep_edges", output_brep_edges)
