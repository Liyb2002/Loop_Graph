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
    def __init__(self, stroke_loop_embeddings, loop_neighboring_vertical, loop_neighboring_horizontal, stroke_to_brep):
        super(SketchHeteroData, self).__init__()

        # Node features
        self['stroke'].x = stroke_loop_embeddings

        # Convert adjacency matrix to edge indices
        edge_index_vertical = self._adjacency_to_edge_index(loop_neighboring_vertical)
        edge_index_horizontal = self._adjacency_to_edge_index(loop_neighboring_horizontal)

        # Set edge indices for vertical and horizontal connections
        self['stroke', 'verticalNeighboring', 'stroke'].edge_index = edge_index_vertical
        self['stroke', 'horizontalNeighboring', 'stroke'].edge_index = edge_index_horizontal

        # Add additional edge information (e.g., stroke to brep)
        self.build_stroke_loop_representation(stroke_to_brep)

    def _adjacency_to_edge_index(self, adjacency_matrix):
        """
        Converts an adjacency matrix to edge indices.
        Args:
            adjacency_matrix (torch.Tensor): A (num_nodes, num_nodes) adjacency matrix.
        Returns:
            edge_index (torch.Tensor): A (2, num_edges) tensor of edge indices.
        """
        edge_index = torch.nonzero(adjacency_matrix, as_tuple=False).t().contiguous()
        return edge_index
    
    def build_stroke_loop_representation(self, stroke_to_brep):
        """
        Builds the stroke-to-brep representation and appends it to the node features.
        Args:
            stroke_to_brep (torch.Tensor): A tensor representing the stroke-to-brep relationships.
        """
        if stroke_to_brep.shape[0] == 0:
            is_disconnected = torch.ones(self['stroke'].x.shape[0], 1, dtype=torch.int)
        else:
            is_disconnected = (stroke_to_brep.sum(dim=1) == 0).int().unsqueeze(1)
        
        # Append the disconnected information to the node features
        self['stroke'].x = torch.cat((self['stroke'].x, is_disconnected), dim=1)

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
        
        # Combine loop neighboring matrices and create edges between loops
        loop_edge_indices = self._create_loop_neighbor_edges(loop_neighboring_vertical, loop_neighboring_horizontal)
        self['loop', 'neighboring', 'loop'].edge_index = torch.tensor(loop_edge_indices, dtype=torch.long)

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

    def _create_loop_neighbor_edges(self, loop_neighboring_vertical, loop_neighboring_horizontal):
        # Combine neighboring tensors
        combined_neighboring = (loop_neighboring_vertical | loop_neighboring_horizontal)
        loop_edge_indices = ([], [])
        
        # Create edges based on the combined neighboring matrix
        num_loops = combined_neighboring.shape[0]
        for i in range(num_loops):
            for j in range(num_loops):
                if combined_neighboring[i, j] == 1:
                    loop_edge_indices[0].append(i)
                    loop_edge_indices[1].append(j)
        
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
    num_operation_counts = 0

    # find the total number of operations
    for i, (_, stroke) in enumerate(stroke_dict.items()):
        for index in stroke.Op_orders:
            if index > num_operation_counts:
                num_operation_counts = index

    # a map that maps stroke_id (e.g 'edge_0_0' to 0)
    stroke_id_to_index = {}

    node_features = np.zeros((num_strokes, 7))
    operations_order_matrix = np.zeros((num_strokes, num_operation_counts+1))


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
        for stroke_op_count in stroke.Op_orders:
            operations_order_matrix[i, stroke_op_count] = 1


    return node_features, operations_order_matrix


def vis_graph(graph):
    pass