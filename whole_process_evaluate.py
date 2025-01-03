
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
        self.data_dirs = [
            os.path.join(self.data_path, d) 
            for d in os.listdir(self.data_path) 
            if os.path.isdir(os.path.join(self.data_path, d))
        ]

        # List of sublist. Each sublist is all the particles in a data piece
        self.data_particles = [
            [
                os.path.join(data_dir, subfolder)
                for subfolder in os.listdir(data_dir)
                if os.path.isdir(os.path.join(data_dir, subfolder))
            ]
            for data_dir in self.data_dirs
        ]

        # all particles
        self.flatted_particle_folders = [
            folder
            for sublist in self.data_particles
            for folder in sublist
        ]

        # Collect all data pieces: (folder_path, file_index)
        self.data_pieces = []
        for folder in self.flatted_particle_folders:
            canvas_dir = os.path.join(folder, 'canvas')
            if os.path.exists(canvas_dir) and os.path.isdir(canvas_dir):
                shape_files = sorted(
                    f for f in os.listdir(canvas_dir) if f.endswith('_eval_info.pkl')
                )
                for shape_file in shape_files:
                    index = int(shape_file.split('_')[0])
                    self.data_pieces.append((folder, index))

        print(f"Total number of data pieces: {len(self.data_pieces)}")

    def __len__(self):
        return len(self.data_pieces)

    def __getitem__(self, idx):
        # Load stroke_node_features from  _eval_info.pkl
        folder, file_index = self.data_pieces[idx]
        canvas_dir = os.path.join(folder, 'canvas')
        shape_file_path = os.path.join(canvas_dir, f"{file_index}_eval_info.pkl")

        with open(shape_file_path, 'rb') as f:
            shape_data = pickle.load(f)
        
        stroke_node_features = shape_data['stroke_node_features']
        

        # Load generated Brep file
        shape_file_path = os.path.join(canvas_dir, f"brep_{file_index}.step")
        edge_features_list, cylinder_features = Preprocessing.SBGCN.brep_read.create_graph_from_step_file(shape_file_path)
        output_brep_edges = Preprocessing.proc_CAD.helper.pad_brep_features(edge_features_list + cylinder_features)


        # Load gt Brep file
        gt_brep_edges = shape_data['gt_brep_edges']
        gt_to_output_same = shape_data['gt_to_output_same']
        output_to_gt_same = shape_data['output_to_gt_same']

        
        # Find high_dist_indices 
        high_dist_indices = shape_data['high_dist_indices']
        dist_vals = shape_data['dist_vals']


        # Load Pre-computed 
        return stroke_node_features, output_brep_edges, gt_brep_edges, gt_to_output_same, output_to_gt_same, high_dist_indices, dist_vals




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
    
    return torch.mean(min_dist_brep_to_stroke), torch.mean(min_dist_stroke_to_brep) + torch.mean(min_dist_brep_to_stroke)



# --------------------- Main Code --------------------- #



# Set up dataloader
dataset = Evaluation_Dataset('program_output')
data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

total_correct = 0
total = 0

for data in tqdm(data_loader, desc="Evaluating CAD Programs"):
    stroke_node_features, output_brep_edges, gt_brep_edges, gt_to_output_same, output_to_gt_same, high_dist_indices, dist_vals= data

    stroke_node_features = stroke_node_features.squeeze(0)
    stroke_node_features = torch.round(stroke_node_features * 10000) / 10000

    output_brep_edges = output_brep_edges.squeeze(0)
    output_brep_edges = torch.round(output_brep_edges * 10000) / 10000

    gt_brep_edges = gt_brep_edges.squeeze(0)
    gt_brep_edges = torch.round(gt_brep_edges * 10000) / 10000


    if output_brep_edges.shape[0] == 0:
        continue
    
    # Covered = the first shape covers the second, if it is 0, then we are on the right track
    # covered_chamfer_dist, whole_chamfer_dist= chamfer_distance(gt_brep_edges, output_brep_edges)
    # Encoders.helper.vis_brep(stroke_node_features)
    # Encoders.helper.vis_brep(output_brep_edges)
    # Encoders.helper.vis_brep(gt_brep_edges)
    print("output_to_gt_same : on the right track", output_to_gt_same)
    # print("output_to_gt_same", output_to_gt_same)


    print("high_dist_indices", high_dist_indices)
    print("dist_vals", dist_vals)
    print("gt_brep_edges", output_brep_edges)
    Encoders.helper.vis_brep_with_indices(output_brep_edges, high_dist_indices)
    Encoders.helper.vis_brep(gt_brep_edges)

    
    total += 1
print(f"Overall Average Accuracy: {total_correct / total:.4f}, with total_correct : {total_correct} and total: {total}")
