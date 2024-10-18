
from torch.utils.data import Dataset, DataLoader
import os
import pickle
from tqdm import tqdm
import torch

import Preprocessing.SBGCN.brep_read
import Preprocessing.proc_CAD.helper
import Encoders.helper

# --------------------- Dataloader for output --------------------- #
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

        # Load gt Brep file
        gt_brep_file_path = os.path.join(self.data_path, data_dir, 'gt_brep.pkl')

        return stroke_node_features, output_brep_edges




# --------------------- Chamfer Distance Computation --------------------- #
def chamfer_distance(stroke_node_features, output_brep_edges):
    """
    Calculates the Chamfer distance between strokes and BREP edges.

    Parameters:
    - stroke_node_features (torch.Tensor): Tensor of shape (num_strokes, 8), where the first 6 values represent
      two 3D points defining a stroke (start and end points), and the 8th value indicates whether to ignore the stroke (1 to ignore).
    - output_brep_edges (torch.Tensor): Tensor of shape (num_brep_edges, 8), where the first 6 values represent
      two 3D points defining a BREP edge (start and end points), and the 8th value indicates whether to ignore the edge (1 to ignore).

    Returns:
    - chamfer_dist (torch.Tensor): Chamfer distance between the strokes and the BREP edges.
    """

    # Filter valid strokes based on the 8th column (ignore strokes where stroke[8] != 0)
    valid_strokes = stroke_node_features[stroke_node_features[:, 7] == 0]
    
    # Extract the start and end 3D points from valid strokes
    stroke_start_points = valid_strokes[:, :3]  # First 3 values
    stroke_end_points = valid_strokes[:, 3:6]  # Next 3 values
    
    # Combine start and end points to get a list of all 3D points
    stroke_points = torch.cat((stroke_start_points, stroke_end_points), dim=0)

    # Filter valid brep edges based on the 8th column (ignore edges where output_brep_edges[8] != 0)
    valid_brep_edges = output_brep_edges[output_brep_edges[:, 7] == 0]
    
    # Extract the start and end 3D points from valid BREP edges
    brep_start_points = valid_brep_edges[:, :3]  # First 3 values
    brep_end_points = valid_brep_edges[:, 3:6]  # Next 3 values
    
    # Combine start and end points to get a list of all 3D points
    brep_points = torch.cat((brep_start_points, brep_end_points), dim=0)

    # Compute Chamfer distance (forward direction: stroke to BREP)
    dist_stroke_to_brep = torch.cdist(stroke_points, brep_points, p=2)
    min_dist_stroke_to_brep = torch.min(dist_stroke_to_brep, dim=1)[0]

    # Compute Chamfer distance (backward direction: BREP to stroke)
    dist_brep_to_stroke = torch.cdist(brep_points, stroke_points, p=2)
    min_dist_brep_to_stroke = torch.min(dist_brep_to_stroke, dim=1)[0]

    # Average Chamfer distance (forward + backward)
    # chamfer_dist = torch.mean(min_dist_stroke_to_brep) + torch.mean(min_dist_brep_to_stroke)
    
    chamfer_dist = torch.mean(min_dist_brep_to_stroke)

    return chamfer_dist

# --------------------- Main Code --------------------- #

# Set up dataloader
dataset = Evaluation_Dataset('program_output')
data_loader = DataLoader(dataset, batch_size=1, shuffle=True)


for data in tqdm(data_loader, desc="Generating CAD Programs"):
    stroke_node_features, output_brep_edges = data
    stroke_node_features = stroke_node_features.squeeze(0)
    stroke_node_features = torch.round(stroke_node_features * 10000) / 10000

    output_brep_edges = output_brep_edges.squeeze(0)
    output_brep_edges = torch.round(output_brep_edges * 10000) / 10000


    chamfer_dist = chamfer_distance(stroke_node_features, output_brep_edges)
    Encoders.helper.vis_brep(stroke_node_features)
    Encoders.helper.vis_brep(output_brep_edges)
    
    print("chamfer_dist", chamfer_dist)

