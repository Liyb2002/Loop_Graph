
import numpy as np
import networkx as nx
from itertools import combinations, permutations

import torch
from collections import Counter

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

    # Get the nodes of the last stroke
    last_start_point = tuple(np.round(last_stroke[:3], 4))
    last_end_point = tuple(np.round(last_stroke[3:], 4))

    # List of nodes excluding those of the last stroke
    nodes = list(G.nodes)
    nodes.remove(last_start_point)
    nodes.remove(last_end_point)

    # Helper function to check if a set of edges forms a valid cycle
    def check_valid_edges(edges):
        point_count = {}
        for edge in edges:
            point_count[edge[0]] = point_count.get(edge[0], 0) + 1
            point_count[edge[1]] = point_count.get(edge[1], 0) + 1
        # A valid cycle has each node exactly twice
        return all(count == 2 for count in point_count.values())

    # Check for valid loops of size 3 (2 nodes + last stroke)
    for group_nodes in combinations(nodes, 2):
        group_with_last = list(group_nodes) + [last_start_point, last_end_point]
        # Check if these nodes can form a valid subgraph
        if nx.is_connected(G.subgraph(group_with_last)):
            # Generate all permutations of the edges
            for perm_edges in permutations(combinations(group_with_last, 2), 3):
                if check_valid_edges(perm_edges):
                    strokes_in_group = [edge_to_stroke_id.get(edge) or edge_to_stroke_id.get((edge[1], edge[0])) for edge in perm_edges]
                    if None not in strokes_in_group:  # Ensure all edges are found in the mapping
                        valid_groups.append(sorted(strokes_in_group))

    # Check for valid loops of size 4 (3 nodes + last stroke)
    for group_nodes in combinations(nodes, 3):
        group_with_last = list(group_nodes) + [last_start_point, last_end_point]
        # Check if these nodes can form a valid subgraph
        if nx.is_connected(G.subgraph(group_with_last)):
            # Generate all permutations of the edges
            for perm_edges in permutations(combinations(group_with_last, 2), 4):
                if check_valid_edges(perm_edges):
                    strokes_in_group = [edge_to_stroke_id.get(edge) or edge_to_stroke_id.get((edge[1], edge[0])) for edge in perm_edges]
                    if None not in strokes_in_group:  # Ensure all edges are found in the mapping
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
        if len(points) == len(group):
            final_groups.append(group)

    return final_groups



def extract_unique_points(sketch_selection_mask, gnn_graph):
    """
    Extract the unique points from the strokes connected to the loop with the highest probability in the selection mask.
    
    Parameters:
    sketch_selection_mask (torch.Tensor): A tensor of shape (num_loops, 1) representing probabilities for selecting loops.
    gnn_graph (HeteroData): The graph containing loop and stroke nodes, and edges representing their relationships.
    
    Returns:
    unique_points (torch.Tensor): A tensor of unique 3D points extracted from the stroke nodes.
    """

    # 1. Find the loop with the highest probability
    max_prob_loop_idx = torch.argmax(sketch_selection_mask).item()

    # 2. Find the stroke nodes connected to this loop node via 'representedBy' edges
    # Edge indices for 'loop' -> 'stroke' are stored in gnn_graph['loop', 'representedBy', 'stroke'].edge_index
    loop_stroke_edges = gnn_graph['loop', 'representedBy', 'stroke'].edge_index
    connected_stroke_indices = loop_stroke_edges[1][loop_stroke_edges[0] == max_prob_loop_idx]  # Get stroke indices for the selected loop

    # 3. Extract points from the connected stroke nodes
    stroke_features = gnn_graph['stroke'].x  # Shape: (num_strokes, 7), first 6 values are the 3D points
    points = []
    for stroke_idx in connected_stroke_indices:
        stroke_feature = stroke_features[stroke_idx]
        point1 = stroke_feature[:3] 
        point2 = stroke_feature[3:6] 
        points.append(point1)
        points.append(point2)

    # 4. Remove duplicate points to get unique points
    points_tensor = torch.stack(points)  
    unique_points_tensor = torch.unique(points_tensor, dim=0)  

    return unique_points_tensor




def get_extrude_amount(gnn_graph, extrude_selection_mask):
    """
    Calculate the extrude amount and direction from the stroke with the highest probability in the extrude_selection_mask.
    
    Parameters:
    gnn_graph (HeteroData): The graph containing stroke nodes and their features.
    extrude_selection_mask (torch.Tensor): A tensor of shape (num_strokes, 1) representing probabilities for selecting strokes.
    
    Returns:
    tuple: (float, list) The stroke length and direction of extrusion for the stroke with the highest probability.
    """

    # 1. Find the stroke with the highest value in extrude_selection_mask
    max_prob_stroke_idx = torch.argmax(extrude_selection_mask).item()

    # 2. Extract stroke node features for the selected stroke
    stroke_features = gnn_graph['stroke'].x  # Shape: (num_strokes, 7), first 6 values are the 3D points
    
    # Get the stroke feature for the stroke with the highest probability
    stroke_feature = stroke_features[max_prob_stroke_idx]
    
    # Extract the two 3D points
    point1 = stroke_feature[:3]
    point2 = stroke_feature[3:6]
    
    # Compute the Euclidean distance (length of the stroke)
    stroke_length = torch.dist(point1, point2).item()
    
    # Compute the direction of the stroke (normalized vector)
    direction_vector = (point2 - point1).tolist()  # Vector from point1 to point2

    return stroke_length, direction_vector


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
