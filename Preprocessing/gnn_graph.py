import os
import torch

import os
import sys
import torch
import torch_geometric
from torch_geometric.data import Data, HeteroData
import numpy as np
import networkx as nx
from matplotlib import pyplot as plt
from networkx.algorithms import community
import seaborn as sns

operations_dict = {     "terminate": 0,
                        "sketch": 1,
                        "extrude": 2,
                        "fillet": 3
                    } 

class SketchHeteroData(HeteroData):
    def __init__(self, stroke_node_features, final_brep_edges):
        super(SketchHeteroData, self).__init__()
        
        # Compute stroke usage
        stroke_used = self._compute_stroke_usage(stroke_node_features, final_brep_edges)
        
        # Build stroke node features with the shape (num_strokes, 7)
        self['stroke'].x = self._build_stroke_features(stroke_node_features, stroke_used)

        # Build connected strokes matrix and connect edges
        connected_strokes = self._build_connected_strokes(stroke_node_features)
        self._add_connect_edges(connected_strokes)

        # Build order edges
        self._add_order_edges(stroke_node_features)

    def _compute_stroke_usage(self, stroke_node_features, final_brep_edges):
        """ 
        Compute a (num_strokes, 1) matrix telling if a stroke is represented in final_brep_edges.
        """

        num_strokes = stroke_node_features.shape[0]
        stroke_used = torch.zeros((num_strokes, 1), dtype=torch.float)  # Initialize as not used

        if final_brep_edges.shape[0] == 0:
            return stroke_used

        # Iterate through each stroke
        for stroke_idx in range(num_strokes):
            stroke_start = stroke_node_features[stroke_idx, :3]  # First 3D point
            stroke_end = stroke_node_features[stroke_idx, 3:6]   # Second 3D point
            
            # Check if this stroke is represented in final_brep_edges
            for brep_edge in final_brep_edges:
                brep_start = brep_edge[:3]  # First 3D point of brep
                brep_end = brep_edge[3:6]   # Second 3D point of brep
                
                # Check if points match (either in the same order or reversed)
                if (torch.allclose(stroke_start, brep_start) and torch.allclose(stroke_end, brep_end)) or \
                   (torch.allclose(stroke_start, brep_end) and torch.allclose(stroke_end, brep_start)):
                    stroke_used[stroke_idx] = 1  # Mark this stroke as used
                    break  # No need to check further once a match is found

        return stroke_used

    def _build_stroke_features(self, stroke_node_features, stroke_used):
        """
        Build the stroke node features for the graph. The shape will be (num_strokes, 7).
        First 6 columns are from stroke_node_features, and the last column is stroke_used.
        """
        # Concatenate the first 6 columns from stroke_node_features with stroke_used (the last column)
        stroke_features = torch.cat([stroke_node_features[:, :6], stroke_used], dim=1)
        return stroke_features

    def _build_connected_strokes(self, stroke_node_features):
        """
        Build the connected strokes matrix. If two strokes share a common point, mark them as connected.
        """
        num_strokes = stroke_node_features.shape[0]
        connected_strokes = torch.zeros((num_strokes, num_strokes), dtype=torch.float)

        # Iterate over pairs of strokes and check if they share a common point
        for i in range(num_strokes):
            stroke_i_start = stroke_node_features[i, :3]
            stroke_i_end = stroke_node_features[i, 3:6]

            for j in range(i + 1, num_strokes):
                stroke_j_start = stroke_node_features[j, :3]
                stroke_j_end = stroke_node_features[j, 3:6]

                # Check if any of the two points match (in either order)
                if (torch.allclose(stroke_i_start, stroke_j_start) or torch.allclose(stroke_i_start, stroke_j_end) or
                    torch.allclose(stroke_i_end, stroke_j_start) or torch.allclose(stroke_i_end, stroke_j_end)):
                    connected_strokes[i, j] = 1
                    connected_strokes[j, i] = 1  # Symmetric for undirected edges

        return connected_strokes

    def _add_connect_edges(self, connected_strokes):
        """
        Add edges to the graph based on the connected_strokes matrix.
        """
        num_strokes = connected_strokes.shape[0]
        edge_index = [[], []]

        # Iterate over the connected_strokes matrix and add edges
        for i in range(num_strokes):
            for j in range(i + 1, num_strokes):  # Only upper triangle to avoid duplicates
                if connected_strokes[i, j] == 1:
                    edge_index[0].append(i)
                    edge_index[1].append(j)
                    edge_index[0].append(j)
                    edge_index[1].append(i)  # Add the reverse edge for undirected graph

        # Convert to tensor and add to the graph
        self['stroke', 'connect', 'stroke'].edge_index = torch.tensor(edge_index, dtype=torch.long)

    def _add_order_edges(self, stroke_node_features):
        """
        Add order edges to the graph based on the order of strokes in stroke_node_features.
        """
        num_strokes = stroke_node_features.shape[0]
        edge_index = [[], []]

        # Iterate over strokes and connect them sequentially
        for i in range(num_strokes - 1):
            edge_index[0].append(i)
            edge_index[1].append(i + 1)

        # Convert to tensor and add to the graph
        self['stroke', 'order', 'stroke'].edge_index = torch.tensor(edge_index, dtype=torch.long)



class SketchLoopGraph(HeteroData):
    def __init__(self, stroke_cloud_loops, stroke_node_features, loop_neighboring_vertical, loop_neighboring_horizontal, loop_neighboring_contained, loop_to_brep):
        super(SketchLoopGraph, self).__init__()

        # Use only the first 6 values of stroke_node_features
        self['stroke'].x = torch.tensor(stroke_node_features[:, :6], dtype=torch.float)

        # Loop node features based on the strokes and loop_to_brep
        self['loop'].x = self._compute_loop_features(stroke_cloud_loops, stroke_node_features, loop_to_brep)

        # Create edges between loops and strokes
        loop_indices, stroke_indices = self._create_loop_stroke_edges(stroke_cloud_loops)
        self['loop', 'representedBy', 'stroke'].edge_index = torch.tensor([loop_indices, stroke_indices], dtype=torch.long)
        
        # Create neighboring_vertical edges
        vertical_edge_indices = self._create_loop_neighbor_edges(loop_neighboring_vertical)
        self['loop', 'neighboring_vertical', 'loop'].edge_index = torch.tensor(vertical_edge_indices, dtype=torch.long)

        # Create neighboring_horizontal edges
        horizontal_edge_indices = self._create_loop_neighbor_edges(loop_neighboring_horizontal)
        self['loop', 'neighboring_horizontal', 'loop'].edge_index = torch.tensor(horizontal_edge_indices, dtype=torch.long)

        # Create directed edges for contained loops
        contained_loop_edges = self._create_contained_edges(loop_neighboring_contained)
        self['loop', 'contains', 'loop'].edge_index = torch.tensor(contained_loop_edges, dtype=torch.long)

        # Create directed edges between loops based on their order
        ordered_loop_edges = self._create_ordered_loop_edges(stroke_cloud_loops)
        self['loop', 'order', 'loop'].edge_index = torch.tensor(ordered_loop_edges, dtype=torch.long)

    def _compute_loop_features(self, stroke_cloud_loops, stroke_node_features, loop_to_brep):
        num_loops = len(stroke_cloud_loops)
        loop_features = []

        # Process each loop
        for loop_idx, loop in enumerate(stroke_cloud_loops):
            # The first 5 values: average of the 7th value from connected strokes
            stroke_seventh_values = [stroke_node_features[stroke_idx, 6] for stroke_idx in loop]
            if len(stroke_seventh_values) > 0:
                avg_value = sum(stroke_seventh_values) / len(stroke_seventh_values)
            else:
                avg_value = 0  # In case of an empty loop
            
            loop_feature = [avg_value] * 5  # Repeat the average for the first 5 values

            # The last value: check the sum of the loop's corresponding row in loop_to_brep
            if loop_to_brep.shape[0] == 0:
                last_value = 0  # If loop_to_brep is empty, set to 0
            else:
                # Compute the last value: 1 if any value in the row is 1, otherwise 0
                last_value = float(loop_to_brep[loop_idx, :].sum() > 0)

            loop_feature.append(last_value)  # Add the last value

            loop_features.append(loop_feature)

        return torch.tensor(loop_features, dtype=torch.float)

    def _create_loop_stroke_edges(self, stroke_cloud_loops):
        loop_indices = []
        stroke_indices = []
        
        # Create edges between each loop node and its corresponding stroke nodes
        for loop_idx, loop in enumerate(stroke_cloud_loops):
            for stroke_idx in loop:
                loop_indices.append(loop_idx)  # Connect the current loop node
                stroke_indices.append(stroke_idx)  # To each stroke node in the sublist
        
        return loop_indices, stroke_indices

    def _create_loop_neighbor_edges(self, loop_neighboring):
        """ Create non-directed edges for neighboring loops """
        loop_edge_indices = ([], [])
        
        num_loops = loop_neighboring.shape[0]
        for i in range(num_loops):
            for j in range(num_loops):
                if loop_neighboring[i, j] == 1:
                    loop_edge_indices[0].append(i)
                    loop_edge_indices[1].append(j)
        
        return loop_edge_indices

    def _create_contained_edges(self, loop_neighboring_contained):
        """ Create directed edges based on loop containment relationships """
        loop_edge_indices = ([], [])

        num_loops = loop_neighboring_contained.shape[0]
        for i in range(num_loops):
            for j in range(num_loops):
                if loop_neighboring_contained[i, j] == 1:  # If loop_i contains loop_j
                    loop_edge_indices[0].append(i)  # Directed edge from i (containing loop)
                    loop_edge_indices[1].append(j)  # To j (contained loop)
        
        return loop_edge_indices

    def _create_ordered_loop_edges(self, stroke_cloud_loops):
        # Compute the average stroke index for each loop
        loop_order = [(idx, sum(loop) / len(loop) if len(loop) > 0 else float('inf')) for idx, loop in enumerate(stroke_cloud_loops)]
        # Sort loops by average stroke index (ascending order)
        loop_order.sort(key=lambda x: x[1])

        # Create directed edges from the first loop to the last one
        ordered_loop_edges = ([], [])
        for i in range(len(loop_order) - 1):
            ordered_loop_edges[0].append(loop_order[i][0])
            ordered_loop_edges[1].append(loop_order[i + 1][0])
        
        return ordered_loop_edges






def build_graph(stroke_dict):
    num_strokes = len(stroke_dict)
    num_operations = 0

    # # find the total number of operations
    # for i, (_, stroke) in enumerate(stroke_dict.items()):
    #     for index in stroke.Op_orders:
    #         if index > num_operation_counts:
    #             num_operation_counts = index

    # # a map that maps stroke_id (e.g 'edge_0_0' to 0)
    # stroke_id_to_index = {}

    for i, key in enumerate(sorted(stroke_dict.keys())):
        stroke = stroke_dict[key]
        if len(stroke.Op) > 0 and num_operations < stroke.Op[0]:
            num_operations = stroke.Op[0]

    node_features = np.zeros((num_strokes, 7))
    operations_order_matrix = np.zeros((num_strokes, num_operations+1))


    for i, key in enumerate(sorted(stroke_dict.keys())):
        stroke = stroke_dict[key]

        # build node_features
        # node_features has shape num_strokes x 6, which is the starting and ending point
        start_point = stroke.vertices[0].position
        end_point = stroke.vertices[1].position
        alpha_value = stroke.alpha_value
        node_features[i, :3] = start_point
        node_features[i, 3:6] = end_point
        node_features[i, 6:] = alpha_value

        # build operation_order_matrix
        # operation_order_matrix has shape num_strokes x num_ops
        for stroke_op_count in stroke.Op:
            operations_order_matrix[i, stroke_op_count] = 1


    return node_features, operations_order_matrix

