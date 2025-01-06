
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

        
        # Find high_dist_indices 
        on_right_track = shape_data['on_right_track']
        is_finished = shape_data['is_finished']
        high_dist_indices = shape_data['high_dist_indices']


        # Load Pre-computed 
        return stroke_node_features, output_brep_edges, gt_brep_edges, on_right_track, is_finished, high_dist_indices



# --------------------- Chamfer Distance Computation --------------------- #
def chamfer_distance_brep(gt_brep_edges, output_brep_edges, threshold=0.05):
    """
    Calculates the maximum Chamfer distance between ground truth (GT) BREP edges and output BREP edges.
    Also identifies indices of output edges with a minimum distance greater than the threshold.

    Parameters:
    - gt_brep_edges (numpy.ndarray or torch.Tensor): Array or tensor of shape (num_gt_edges, 10),
      where the first 6 values represent two 3D points defining a GT BREP edge (start and end points).
    - output_brep_edges (numpy.ndarray or torch.Tensor): Array or tensor of shape (num_output_edges, 10),
      where the first 6 values represent two 3D points defining an output BREP edge (start and end points).
    - threshold (float): Distance threshold to identify output edges with high minimum distance.

    Returns:
    - max_dist_gt_to_output (torch.Tensor): Maximum Chamfer distance from GT edges to output edges.
    - max_dist_output_to_gt (torch.Tensor): Maximum Chamfer distance from output edges to GT edges.
    - high_dist_indices (list): List of indices of output edges with minimum distance > threshold.
    """
    # Ensure inputs are tensors
    if not isinstance(gt_brep_edges, torch.Tensor):
        gt_brep_edges = torch.tensor(gt_brep_edges, dtype=torch.float32)
    if not isinstance(output_brep_edges, torch.Tensor):
        output_brep_edges = torch.tensor(output_brep_edges, dtype=torch.float32)

    # Extract start and end points for valid GT edges
    gt_start_points = gt_brep_edges[:, :3]  # First 3 values
    gt_end_points = gt_brep_edges[:, 3:6]  # Next 3 values
    gt_points = torch.cat((gt_start_points, gt_end_points), dim=0)  # Combine start and end points

    # Extract start and end points for valid output edges
    output_start_points = output_brep_edges[:, :3]  # First 3 values
    output_end_points = output_brep_edges[:, 3:6]  # Next 3 values
    output_points = torch.cat((output_start_points, output_end_points), dim=0)  # Combine start and end points

    # Compute Chamfer distance (GT to Output)
    dist_gt_to_output = torch.cdist(gt_points, output_points, p=2)  # Pairwise distances
    min_dist_gt_to_output = torch.min(dist_gt_to_output, dim=1)[0]  # Minimum distance for each GT point
    max_dist_gt_to_output = torch.max(min_dist_gt_to_output)  # Maximum distance

    # Compute Chamfer distance (Output to GT)
    dist_output_to_gt = torch.cdist(output_points, gt_points, p=2)  # Pairwise distances
    min_dist_output_to_gt = torch.min(dist_output_to_gt, dim=1)[0]  # Minimum distance for each Output point
    max_dist_output_to_gt = torch.max(min_dist_output_to_gt)  # Maximum distance

    # Identify indices of output edges with high-distance points
    high_dist_point_indices = torch.where(min_dist_output_to_gt > threshold)[0]  # Indices of high-dist points

    # Map high-distance points back to edges
    num_output_edges = gt_brep_edges.shape[0]
    high_dist_edge_indices = set()
    dist_vals = []
    edge_to_max_dist = {}

    for point_idx in high_dist_point_indices:
        # Each edge contributes two points in output_points (start and end)
        edge_idx = point_idx // 2  # Integer division to map back to edge index
        if edge_idx < num_output_edges:
            high_dist_edge_indices.add(edge_idx)
            # Update the maximum distance for this edge
            edge_to_max_dist[edge_idx] = max(edge_to_max_dist.get(edge_idx, 0), min_dist_output_to_gt[point_idx].item())

    # Sort edges and prepare dist_vals
    high_dist_edge_indices = sorted(list(high_dist_edge_indices))
    dist_vals = [edge_to_max_dist[edge_idx] for edge_idx in high_dist_edge_indices]

    return max_dist_gt_to_output, max_dist_output_to_gt, high_dist_edge_indices, dist_vals



def brep_difference(prev_brep_edges, new_brep_edges):
    """
    Returns the edges in new_brep_edges that are not present in prev_brep_edges.

    Two edges are considered the same if they have the same two points [:3] and [3:6],
    regardless of the order of the points.

    Parameters:
    - prev_brep_edges (numpy.ndarray or torch.Tensor): Array or tensor of shape (num_prev_edges, 10),
      where the first 6 values represent two 3D points defining an edge (start and end points).
    - new_brep_edges (numpy.ndarray or torch.Tensor): Array or tensor of shape (num_new_edges, 10),
      where the first 6 values represent two 3D points defining an edge (start and end points).

    Returns:
    - unique_new_edges (torch.Tensor): Tensor of edges in new_brep_edges that are not in prev_brep_edges.
    """

    if prev_brep_edges is None:
        return new_brep_edges
    
    # Ensure inputs are tensors
    if isinstance(prev_brep_edges, torch.Tensor):
        prev_brep_edges = prev_brep_edges.cpu().numpy()
    if isinstance(new_brep_edges, torch.Tensor):
        new_brep_edges = new_brep_edges.cpu().numpy()

    # Helper function to normalize edges (sort start and end points)
    def normalize_edge(edge):
        start, end = edge[:3], edge[3:6]
        return tuple(sorted([tuple(start), tuple(end)]))

    # Normalize and create sets of edges
    prev_edge_set = {normalize_edge(edge) for edge in prev_brep_edges}
    unique_edges = []

    for edge in new_brep_edges:
        if normalize_edge(edge) not in prev_edge_set:
            unique_edges.append(edge)

    # Convert the unique edges back to a tensor
    unique_new_edges = torch.tensor(unique_edges, dtype=torch.float32)

    return unique_new_edges

# --------------------- Main Code --------------------- #



# Set up dataloader
dataset = Evaluation_Dataset('program_output')
data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

total_correct = 0
total = 0

prev_brep_edges = None

for data in tqdm(data_loader, desc="Evaluating CAD Programs"):
    stroke_node_features, output_brep_edges, gt_brep_edges, on_right_track, is_finished, high_dist_indices = data

    stroke_node_features = stroke_node_features.squeeze(0)
    stroke_node_features = torch.round(stroke_node_features * 10000) / 10000

    output_brep_edges = output_brep_edges.squeeze(0)
    output_brep_edges = torch.round(output_brep_edges * 10000) / 10000

    gt_brep_edges = gt_brep_edges.squeeze(0)
    gt_brep_edges = torch.round(gt_brep_edges * 10000) / 10000


    print("on_right_track ?", on_right_track)
    print("is_finished?", is_finished)

    high_dist_indices_values = [idx.item() for tensor in high_dist_indices for idx in tensor]    

    # if not on_right_track:
    #     Encoders.helper.vis_brep_with_indices(output_brep_edges, high_dist_indices)

    if is_finished:
        print("output_brep_edges", output_brep_edges.shape)
        Encoders.helper.vis_brep(output_brep_edges)

    # unique_new_edges = brep_difference(prev_brep_edges, output_brep_edges)
    # Encoders.helper.vis_brep(unique_new_edges)

    
    total += 1
print(f"Overall Average Accuracy: {total_correct / total:.4f}, with total_correct : {total_correct} and total: {total}")
