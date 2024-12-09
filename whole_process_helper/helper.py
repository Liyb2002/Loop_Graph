
import numpy as np
import networkx as nx
from itertools import combinations, permutations

import torch
from collections import Counter
import os
import shutil
import particle

import torch.nn.functional as F


from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.StlAPI import StlAPI_Writer
from OCC.Core.BRep import BRep_Builder
from OCC.Core.TopoDS import TopoDS_Compound

# --------------------------------------------------------------------------- #

def face_aggregate_addStroke(stroke_matrix):
    """
    This function finds valid loops of strokes with size 3 or 4 using NetworkX, ensuring that each loop
    contains the last stroke from the matrix. Only groups involving the last stroke are considered.
    
    Parameters:
    stroke_matrix (numpy.ndarray): A matrix of shape (num_strokes, 7) where each row represents a stroke
                                   with start and end points in 3D space.

    Returns:
    list: A list of indices of valid loops of strokes, where each loop contains either 3 or 4 strokes,
          and every group includes the last row of the stroke matrix.
    """
    
    # Check if there are fewer than 4 strokes
    if stroke_matrix.shape[0] < 4:
        return []

    # Ensure input is a numpy array and ignore the last column
    stroke_matrix = np.array(stroke_matrix)[:, :6]
    
    # Split the matrix into the last stroke and the rest
    last_stroke = stroke_matrix[-1]
    rest_strokes = stroke_matrix[:-1]
    
    # Initialize the graph
    G = nx.Graph()
    
    # Add edges to the graph based on strokes and store the edge-to-stroke mapping
    edge_to_stroke_id = {}
    for idx, stroke in enumerate(stroke_matrix):
        start_point = tuple(np.round(stroke[:3], 4))
        end_point = tuple(np.round(stroke[3:], 4))
        G.add_edge(start_point, end_point)
        # Store both directions in the dictionary to handle undirected edges
        edge_to_stroke_id[(start_point, end_point)] = idx
        edge_to_stroke_id[(end_point, start_point)] = idx  # Add both directions for undirected graph

    # List to store valid groups
    valid_groups = []

    # Get the nodes (points) of the last stroke
    last_start_point = tuple(np.round(last_stroke[:3], 4))
    last_end_point = tuple(np.round(last_stroke[3:], 4))

    # List of nodes excluding those of the last stroke
    nodes = list(G.nodes)
    nodes.remove(last_start_point)
    nodes.remove(last_end_point)

    # Helper function to check if a set of strokes forms a valid cycle
    def check_valid_strokes(strokes):
        point_count = {}
        for stroke_idx in strokes:
            stroke = stroke_matrix[stroke_idx]
            start_point = tuple(stroke[:3])
            end_point = tuple(stroke[3:])
            point_count[start_point] = point_count.get(start_point, 0) + 1
            point_count[end_point] = point_count.get(end_point, 0) + 1
        # A valid cycle has each point exactly twice
        return all(count == 2 for count in point_count.values())

    # Check for valid loops of size 3 (3 strokes, including the last one)
    for group_nodes in combinations(nodes, 1):  # Find one additional point (last stroke gives 2 points)
        group_with_last = [last_start_point, last_end_point] + list(group_nodes)
        # Find all possible combinations of strokes that connect these 3 points
        for perm_edges in permutations(combinations(group_with_last, 2), 3):
            strokes_in_group = [edge_to_stroke_id.get(edge) or edge_to_stroke_id.get((edge[1], edge[0])) for edge in perm_edges]
            if None not in strokes_in_group and check_valid_strokes(strokes_in_group):
                if edge_to_stroke_id[(last_start_point, last_end_point)] in strokes_in_group:
                    valid_groups.append(sorted(strokes_in_group))

    # Check for valid loops of size 4 (4 strokes, including the last one)
    for group_nodes in combinations(nodes, 2):  # Find two additional points (last stroke gives 2 points)
        group_with_last = [last_start_point, last_end_point] + list(group_nodes)
        # Find all possible combinations of strokes that connect these 4 points
        for perm_edges in permutations(combinations(group_with_last, 2), 4):
            strokes_in_group = [edge_to_stroke_id.get(edge) or edge_to_stroke_id.get((edge[1], edge[0])) for edge in perm_edges]
            if None not in strokes_in_group and check_valid_strokes(strokes_in_group):
                if edge_to_stroke_id[(last_start_point, last_end_point)] in strokes_in_group:
                    valid_groups.append(sorted(strokes_in_group))

    # Remove duplicate loops by converting to a set of frozensets
    unique_groups = list(set(frozenset(group) for group in valid_groups))

    # Final check: Ensure each group has the same number of unique points as edges
    final_groups = []
    for group in unique_groups:
        points = set()
        for edge_id in group:
            stroke = stroke_matrix[edge_id]
            points.add(tuple(stroke[:3]))
            points.add(tuple(stroke[3:]))
        if len(points) == len(group):  # A valid cycle should have exactly len(group) + 1 unique points
            final_groups.append(group)

    return final_groups



# --------------------------------------------------------------------------- #

def reorder_strokes_to_neighbors(strokes):
    """
    Reorder strokes so that they form a continuous loop of connected points.
    
    Parameters:
    strokes (list): A list of strokes, where each stroke is a tuple (A, B) representing two points.
    
    Returns:
    ordered_points (torch.Tensor): A tensor of ordered points forming a continuous loop.
    """
    # Start with the first stroke (A, B)
    first_stroke = strokes[0]
    ordered_points = [first_stroke[0], first_stroke[1]]
    
    remaining_strokes = strokes[1:]

    # Continue until we find the stroke that brings us back to the first point
    while remaining_strokes:
        last_point = ordered_points[-1]  # The last point in the ordered list

        for i, stroke in enumerate(remaining_strokes):
            pointA, pointB = stroke
            
            # Check if last_point is either pointA or pointB of the current stroke
            if last_point.equal(pointA):  # If last_point matches pointA
                ordered_points.append(pointB)  # Add pointB to the list
                remaining_strokes.pop(i)  # Remove this stroke from the list
                break
            elif last_point.equal(pointB):  # If last_point matches pointB
                ordered_points.append(pointA)  # Add pointA to the list
                remaining_strokes.pop(i)  # Remove this stroke from the list
                break

        # Stop if we encounter the first point again
        if ordered_points[-1].equal(ordered_points[0]):
            break
    
    ordered_points.pop()

    return torch.stack(ordered_points)  # Convert the list of points back to a tensor


def extract_unique_points(max_prob_loop_idx, gnn_graph):
    """
    Extract strokes from the loop with the highest probability in the selection mask and reorder them.
    
    Parameters:
    sketch_selection_mask (torch.Tensor): A tensor of shape (num_loops, 1) representing probabilities for selecting loops.
    gnn_graph (HeteroData): The graph containing loop and stroke nodes, and edges representing their relationships.
    
    Returns:
    ordered_points (torch.Tensor): A tensor of ordered points forming a continuous loop.
    """


    # 2. Find the stroke nodes connected to this loop node via 'representedBy' edges
    loop_stroke_edges = gnn_graph['loop', 'represented_by', 'stroke'].edge_index
    connected_stroke_indices = loop_stroke_edges[1][loop_stroke_edges[0] == max_prob_loop_idx]

    if connected_stroke_indices.shape[0] == 1:
        circle_stroke = gnn_graph['stroke'].x[connected_stroke_indices[0]]
        return circle_stroke.unsqueeze(0)

    # 3. Extract strokes (pairs of points) from the connected stroke nodes
    stroke_features = gnn_graph['stroke'].x  # Shape: (num_strokes, 7), first 6 values are the 3D points
    strokes = []
    for stroke_idx in connected_stroke_indices:
        stroke_feature = stroke_features[stroke_idx]
        pointA = stroke_feature[:3]  # First point of the stroke
        pointB = stroke_feature[3:6]  # Second point of the stroke
        strokes.append((pointA, pointB))  # Store as a tuple (A, B)

    # 4. Reorder the strokes to form a continuous loop
    ordered_points_tensor = reorder_strokes_to_neighbors(strokes)

    return ordered_points_tensor



# --------------------------------------------------------------------------- #


def get_extrude_amount(gnn_graph, extrude_selection_mask, sketch_points, brep_edges):
    """
    Calculate the extrude amount and direction from the stroke with the highest probability in the extrude_selection_mask.
    The extrusion direction is determined by identifying which point of the extruding stroke lies on the same plane
    as the sketch_points (coplanar points). If brep_edges are provided, the function determines whether the extrusion 
    is going into or out of the brep, and adjusts the stroke length accordingly.

    Parameters:
    gnn_graph (HeteroData): The graph containing stroke nodes and their features.
    extrude_selection_mask (torch.Tensor): A tensor of shape (num_strokes, 1) representing probabilities for selecting strokes.
    sketch_points (list): A list of points that are coplanar, meaning they share the same value along one axis.
    brep_edges (torch.Tensor): A tensor of shape (num_strokes, 6) representing the brep edges.

    Returns:
    tuple: (float, list) The stroke length and direction of extrusion for the stroke with the highest probability.
    """

    # 1. Find the stroke with the highest value in extrude_selection_mask
    top3_vals, top3_idxs = torch.topk(extrude_selection_mask.view(-1), 3)
    total_sum = top3_vals.sum()
    relative_probs = top3_vals / total_sum
    sampled_idx = torch.multinomial(relative_probs, 1).item()

    selected_idx = top3_idxs[sampled_idx].item()
    selected_prob = top3_vals[sampled_idx].item()




    # 2. Extract stroke node features for the selected stroke
    stroke_features = gnn_graph['stroke'].x  # Shape: (num_strokes, 7), first 6 values are the 3D points
    stroke_feature = stroke_features[selected_idx]



    if sketch_points.shape[0] == 1:
        circle_stroke = sketch_points.squeeze(0)
        # circle stroke
        normal_vector = circle_stroke[3:6]

        # Find common_axis_idx where the normal vector has a value of 1
        common_axis_idx = -1
        for i in range(3):
            if normal_vector[i] == 1 or normal_vector[i] == -1:
                common_axis_idx = i
                break
        
        if common_axis_idx != -1:
            plane_value = circle_stroke[common_axis_idx]
        else:
            plane_value = None  # Handle case if no axis has a value of 1

    else: 
        # 3. Determine the common axis and value from the coplanar sketch_points
        sketch_points_tensor = sketch_points.clone().detach().float()  # Convert to tensor if needed and clone
        common_axes = (torch.all(sketch_points_tensor[:, 0] == sketch_points_tensor[0, 0]),
                    torch.all(sketch_points_tensor[:, 1] == sketch_points_tensor[0, 1]),
                    torch.all(sketch_points_tensor[:, 2] == sketch_points_tensor[0, 2]))

        # Find the index of the common axis (x: 0, y: 1, z: 2)
        common_axis_idx = common_axes.index(True)
        plane_value = sketch_points_tensor[0, common_axis_idx]



    # Extract the two 3D points for the stroke
    point1 = stroke_feature[:3]
    point2 = stroke_feature[3:6]

    # 4. Determine which point of the extruding stroke is on the same plane as the sketch_points
    if torch.isclose(point1[common_axis_idx], plane_value):
        start_point, end_point = point1, point2
    else:
        start_point, end_point = point2, point1

    # 5. Compute the length of the stroke
    stroke_length = torch.dist(start_point, end_point).item()
    
    # 6. Compute the direction of the extrusion (from start_point to end_point)
    direction_vector = (end_point - start_point).tolist()

    # If brep_edges is empty, return the stroke length as positive
    if brep_edges.shape[0] == 0:
        return stroke_length, direction_vector

    # 7. Iterate through each brep_edge to find the first one that satisfies the conditions
    brep_edges_tensor = torch.tensor(brep_edges, dtype=torch.float32)

    selected_brep_edge = None
    on_axis_point = None
    other_point = None

    for brep_edge in brep_edges_tensor:
        brep_point1 = brep_edge[:3]
        brep_point2 = brep_edge[3:6]

        # Check if one of the brep_edge points is on the same plane (has the same value on the common axis)
        if torch.isclose(brep_point1[common_axis_idx], plane_value) and not torch.isclose(brep_point2[common_axis_idx], plane_value):
            selected_brep_edge = brep_edge
            on_axis_point = brep_point1
            other_point = brep_point2
            break
        elif torch.isclose(brep_point2[common_axis_idx], plane_value) and not torch.isclose(brep_point1[common_axis_idx], plane_value):
            selected_brep_edge = brep_edge
            on_axis_point = brep_point2
            other_point = brep_point1
            break

    # 8. If we found a matching brep edge, compute the brep direction
    if selected_brep_edge is not None:
        # Compute the brep direction from the other_point to the on_axis_point
        brep_direction = (other_point - on_axis_point).tolist()

        # 9. Check if brep_direction is opposite to direction_vector
        dot_product = sum(brep_direction[i] * direction_vector[i] for i in range(3))

        # If directions are opposite (dot product < 0), stroke_length should be positive, otherwise negative
        stroke_length = abs(stroke_length) if dot_product < 0 else -abs(stroke_length)

    return stroke_length, direction_vector, selected_prob



def extrude_strokes(gnn_graph, extrude_selection_mask):
    """
    Outputs the stroke features of all selected strokes in the extrude_selection_mask.
    
    Parameters:
    gnn_graph (HeteroData): The graph containing stroke nodes and their features.
    extrude_selection_mask (torch.Tensor): A tensor of shape (num_strokes, 1) representing probabilities for selecting strokes.
    
    Returns:
    torch.Tensor: A tensor containing the features of the selected strokes.
    """

    # 1. Select stroke nodes with prob > 0.5
    max_prob_stroke_idx = torch.argmax(extrude_selection_mask).item()

    # 2. Extract stroke features for the selected stroke indices
    stroke_features = gnn_graph['stroke'].x  # Shape: (num_strokes, 7), first 6 values are the 3D points
    
    # 3. Get the features for the selected strokes
    selected_stroke_feature = stroke_features[max_prob_stroke_idx]

    return selected_stroke_feature



def clean_mask(sketch_selection_mask, selected_loop_idx):
    # Create a tensor of zeros with the same shape as sketch_selection_mask
    cleaned_mask = torch.zeros_like(sketch_selection_mask)

    # Set the row with the highest probability to 1
    cleaned_mask[selected_loop_idx] = 1

    return cleaned_mask
    
# --------------------------------------------------------------------------- #


def get_fillet_amount(gnn_graph, fillet_selection_mask, brep_edges):

    top2_vals, top2_idxs = torch.topk(fillet_selection_mask.view(-1), 1)
    total_sum = top2_vals.sum()
    relative_probs = top2_vals / total_sum
    sampled_idx = torch.multinomial(relative_probs, 1).item()

    selected_idx = top2_idxs[sampled_idx].item()
    selected_prob = top2_vals[sampled_idx].item()

    stroke_features = gnn_graph['stroke'].x  # Shape: (num_strokes, 7), first 6 values are the 3D points
    fillet_stroke = stroke_features[selected_idx]
    


    # Step 1: Extract all unique 3D points from chamfer_strokes
    point1 = fillet_stroke[:3]
    point2 = fillet_stroke[3:6]


    # Convert brep_edges to a PyTorch tensor
    if isinstance(brep_edges, (np.ndarray, list)):
        brep_edges = torch.tensor(brep_edges, dtype=point1.dtype)


    min_distance = 100
    fillet_edge = None

    # Step 2 and 3: Iterate over brep_edges to find the matching edge
    for edge in brep_edges:
        edge_point1 = edge[:3]
        edge_point2 = edge[3:6]
        edge_mid_point = (edge_point1 + edge_point2) / 2

        # Compute distances from edge_mid_point to all fillet_points
        distance1 = torch.norm(point1 - edge_mid_point)
        distance2 = torch.norm(point2 - edge_mid_point)

        # Check if all distances are the same within a small tolerance
        if torch.allclose(distance1, distance2, atol=1e-2):
            
            if distance1 < min_distance:
                min_distance = distance1
                fillet_edge = edge


    if fillet_edge is not None:
        # Compute chamfer_amount
        example_point = fillet_stroke[0:3]
        example_center = fillet_stroke[7:10]

        dist = torch.norm(example_point - example_center)
        return fillet_edge, dist, selected_prob

    return None, None, 0


# --------------------------------------------------------------------------- #



def get_chamfer_amount(gnn_graph, chamfer_selection_mask, brep_edges):
    
    top2_vals, top2_idxs = torch.topk(chamfer_selection_mask.view(-1), 1)
    total_sum = top2_vals.sum()
    relative_probs = top2_vals / total_sum
    sampled_idx = torch.multinomial(relative_probs, 1).item()

    selected_idx = top2_idxs[sampled_idx].item()
    selected_prob = top2_vals[sampled_idx].item()

    stroke_features = gnn_graph['stroke'].x  # Shape: (num_strokes, 7), first 6 values are the 3D points
    chamfer_stroke = stroke_features[selected_idx]


    # Step 1: Extract all unique 3D points from chamfer_strokes
    point1 = chamfer_stroke[:3]
    point2 = chamfer_stroke[3:6]

    # Convert brep_edges to a PyTorch tensor
    if isinstance(brep_edges, (np.ndarray, list)):
        brep_edges = torch.tensor(brep_edges, dtype=point1.dtype)

    # Step 2 and 3: Iterate over brep_edges to find the matching edge
    min_distance = 100
    chamfer_edge = None
    for edge in brep_edges:
        edge_point1 = edge[:3]
        edge_point2 = edge[3:6]
        edge_mid_point = (edge_point1 + edge_point2) / 2

        # Compute distances from edge_mid_point to all chamfer_points
        distance1 = torch.norm(point1 - edge_mid_point)
        distance2 = torch.norm(point2 - edge_mid_point)

        # Check if all distances are the same within a small tolerance
        if torch.allclose(distance1, distance2, atol=1e-2):
            if distance1 < min_distance:
                min_distance = distance1
                chamfer_edge = edge


    if chamfer_edge is not None:
        dist1 = torch.norm(edge[:3] - point1)
        dist2 = torch.norm(edge[3:6] - point1)

        chamfer_amount = min(dist1.item(), dist2.item())
        return edge, chamfer_amount, selected_prob

    return None, None, 0



    
# --------------------------------------------------------------------------- #

def padd_program(past_program):
    """
    Pads the input program token list to a length of 20 with the value 10, 
    and then reshapes it to have a batch size of 1.

    Args:
        past_program (list or torch.Tensor): The input program token list or tensor.

    Returns:
        torch.Tensor: The padded program with a batch size of 1.
    """
    # Convert to tensor if it's a list
    if isinstance(past_program, list):
        past_program = torch.tensor(past_program, dtype=torch.int64)
    
    # Padding the input program to length 20 with the value 10
    pad_size = 20 - past_program.shape[0]
    if pad_size > 0:
        pad = torch.full((pad_size,), 10, dtype=torch.int64, device=past_program.device)
        past_program = torch.cat((past_program, pad), dim=0)
    
    # Reshape to (1, 20) for batch size of 1
    past_program = past_program.unsqueeze(0)  # Adding batch dimension
    
    return past_program



# --------------------------------------------------------------------------- #
def find_valid_sketch(gnn_graph, sketch_selection_mask):
    """
    This function finds the index of the first valid sketch selection by ranking all indices in
    sketch_selection_mask based on their values and checking the corresponding loop node values
    in gnn_graph. It returns the index of the first valid loop node with value 0.

    Parameters:
    gnn_graph (HeteroData): The graph containing loop node features.
    sketch_selection_mask (torch.Tensor): A tensor representing the mask for sketch selection.

    Returns:
    int: The index of the first valid sketch where the loop node value is 0.
         If no valid sketch is found, returns -1.
    """
    
    # Get the indices sorted by the values in sketch_selection_mask (from largest to smallest)
    sorted_indices = torch.argsort(sketch_selection_mask.squeeze(), descending=True)
    valid_indices = []

    # Iterate over the sorted indices and check corresponding loop node values
    for idx in sorted_indices:
        idx = idx.item()  # Convert to Python int

        # Access the value of the loop node at the current index
        loop_node_value = gnn_graph['loop'].x[idx][0].item()  # Assuming loop_node is a single value

        # Check if the loop node value is 0
        if loop_node_value == 0:
            valid_indices.append(idx)

        if len(valid_indices) == 3:
            break

    if len(valid_indices) == 0:
        return [-1], -1

    top_probs = sketch_selection_mask[valid_indices]
    normalized_probs = top_probs / top_probs.sum()
    
    # Sample an index based on the normalized probabilities
    sampled_index = torch.multinomial(normalized_probs, num_samples=1)[0].item() 
    final_index = valid_indices[sampled_index]
    final_prob = max(normalized_probs[sampled_index].item(), top_probs[sampled_index].item())
    
    return [final_index], final_prob


# --------------------------------------------------------------------------- #


def sample_operation(operation_predictions):
    logits_subset = operation_predictions[:, 1:5].squeeze(0)

    
    # Apply softmax to convert logits into probabilities
    probabilities = F.softmax(logits_subset, dim=0)
    
    # Sample an index from the probabilities
    sampled_index = torch.multinomial(probabilities, num_samples=1)
    sampled_class_prob = probabilities[sampled_index].item()
    
    # Map back to the original class indices (1-5)
    sampled_class = sampled_index.item() + 1
    
    return sampled_class, sampled_class_prob



# --------------------------------------------------------------------------- #

def sample_program_termination(stroke_nodes, feature_stroke_mask):

    # 2) Find all feature strokes with mask values > 0.5
    num_feature_strokes = (feature_stroke_mask > 0.5).sum().item()

    # 3) Count used feature strokes among valid strokes
    used_feature_strokes = 0.0
    untouched_feature_idx = []
    for i in range(0, feature_stroke_mask.shape[0]):
        stroke_node = stroke_nodes[i]

        if stroke_node[-1] == 1:
            used_feature_strokes += 1.0

        elif feature_stroke_mask[i] > 0.5:
            untouched_feature_idx.append(i)
    
    termination_prob = used_feature_strokes / num_feature_strokes

    if termination_prob < 0.8:
        termination_prob = 0

    return termination_prob, untouched_feature_idx



# --------------------------------------------------------------------------- #


def brep_to_stl_and_copy(gt_brep_file_path, output_dir, cur_brep_file_path):
    try:
        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Define paths for output files
        output_stl_path = os.path.join(output_dir, 'converted_brep.stl')
        gt_brep_copy_path = os.path.join(output_dir, 'gt_brep.step')

        # Copy the ground truth BREP file
        shutil.copy(gt_brep_file_path, gt_brep_copy_path)

        # Read the current BREP file and convert it to STL
        step_reader = STEPControl_Reader()
        status = step_reader.ReadFile(cur_brep_file_path)

        if status != 1:
            raise ValueError(f"Error: Failed to read the BREP file at {cur_brep_file_path}")

        # Transfer the contents of the BREP file
        step_reader.TransferRoots()
        shape = step_reader.OneShape()

        # Triangulate the shape for STL export
        BRepMesh_IncrementalMesh(shape, 0.1)  # Adjust precision as needed

        # Create a compound to store the shape
        builder = BRep_Builder()
        compound = TopoDS_Compound()
        builder.MakeCompound(compound)
        builder.Add(compound, shape)

        # Write the triangulated shape to an STL file
        stl_writer = StlAPI_Writer()
        stl_writer.SetASCIIMode(False)  # Save as binary for smaller size
        stl_writer.Write(compound, output_stl_path)

    except Exception as e:
        print(f"An error occurred: {e}")


# --------------------------------------------------------------------------- #

def resample_particles(particle_list):
    can_process_particles = []
    success_terminate_particles = []

    for cur_particle in particle_list:
        if cur_particle.valid_particle:
            can_process_particles.append(cur_particle)
        if cur_particle.success_terminate:  
            success_terminate_particles.append(cur_particle)
    
    print("particle_list", len(particle_list))
    print("len can_process_particles", len(can_process_particles))
    print("len success_terminate_particles", len(success_terminate_particles))
