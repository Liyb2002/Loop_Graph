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


class SketchLoopGraph(HeteroData):
    def __init__(self, stroke_cloud_loops, stroke_node_features, connected_stroke_nodes, loop_neighboring_vertical, loop_neighboring_horizontal, loop_neighboring_contained, loop_neighboring_coplanar, loop_to_brep):
        super(SketchLoopGraph, self).__init__()

        # Use all 7 values of stroke_node_features
        self['stroke'].x = torch.tensor(stroke_node_features, dtype=torch.float)

        # Loop node features based on whether the loop is used (repeated 7 times)
        self['loop'].x = self._compute_loop_features(stroke_cloud_loops, loop_to_brep)

        # Create edges between loops and strokes
        loop_stroke_edges = self._create_loop_stroke_edges(stroke_cloud_loops)
        self['loop', 'representedBy', 'stroke'].edge_index = torch.tensor(loop_stroke_edges, dtype=torch.long)
        
        # Create neighboring_vertical edges
        vertical_edge_indices = self._create_loop_neighbor_edges(loop_neighboring_vertical)
        self['loop', 'neighboring_vertical', 'loop'].edge_index = torch.tensor(vertical_edge_indices, dtype=torch.long)

        # Create neighboring_horizontal edges
        horizontal_edge_indices = self._create_loop_neighbor_edges(loop_neighboring_horizontal)
        self['loop', 'neighboring_horizontal', 'loop'].edge_index = torch.tensor(horizontal_edge_indices, dtype=torch.long)

        # Create directed edges for contained loops
        contained_loop_edges = self._create_loop_neighbor_edges(loop_neighboring_contained)
        self['loop', 'contains', 'loop'].edge_index = torch.tensor(contained_loop_edges, dtype=torch.long)

        # Create coplanar edges
        coplanar_edge_indices = self._create_loop_neighbor_edges(loop_neighboring_coplanar)
        self['loop', 'coplanar', 'loop'].edge_index = torch.tensor(coplanar_edge_indices, dtype=torch.long)

        # Create stroke order edges
        stroke_order_edges = self._create_stroke_order_edges(stroke_node_features)
        self['stroke', 'order', 'stroke'].edge_index = torch.tensor(stroke_order_edges, dtype=torch.long)

        # Create stroke connect edges from connected_stroke_nodes
        stroke_connect_edges = self._create_stroke_connect_edges(connected_stroke_nodes)
        self['stroke', 'connect', 'stroke'].edge_index = torch.tensor(stroke_connect_edges, dtype=torch.long)

    def _compute_loop_features(self, stroke_cloud_loops, loop_to_brep):
        """
        Compute loop features. If the loop is used (based on loop_to_brep), assign 1, otherwise 0.
        Repeat the value 7 times to match the feature size of 7.
        """
        num_loops = len(stroke_cloud_loops)
        loop_features = []

        # Process each loop
        for loop_idx in range(num_loops):
            # Check if the loop is used or not
            if loop_to_brep.shape[0] == 0:
                last_value = 0  # If loop_to_brep is empty, set to 0
            else:
                # Compute the last value: 1 if any value in the row is 1, otherwise 0
                last_value = float(loop_to_brep[loop_idx, :].sum() > 0)

            # Repeat the last_value 7 times for the loop feature
            loop_feature = [last_value] * 7
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
        
        return [loop_indices, stroke_indices]

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


    def _create_stroke_order_edges(self, stroke_node_features):
        """
        Create directed edges between strokes based on their order in stroke_node_features.
        """
        num_strokes = stroke_node_features.shape[0]
        edge_index = [[], []]

        # Create order edges by connecting stroke i to stroke i+1
        for i in range(num_strokes - 1):
            edge_index[0].append(i)
            edge_index[1].append(i + 1)

        return edge_index

    def _create_stroke_connect_edges(self, connected_stroke_nodes):
        """
        Create undirected edges between strokes based on the connected_stroke_nodes matrix.
        
        Parameters:
        connected_stroke_nodes (np.ndarray or torch.Tensor): A matrix of shape (num_strokes, num_strokes)
                                                             where [i, j] = 1 if stroke i and stroke j are connected.
        """
        num_strokes = connected_stroke_nodes.shape[0]
        edge_index = [[], []]

        # Iterate over the connected_stroke_nodes matrix
        for i in range(num_strokes):
            for j in range(i + 1, num_strokes):  # Only consider upper triangle to avoid duplicates
                if connected_stroke_nodes[i, j] == 1:
                    edge_index[0].append(i)
                    edge_index[1].append(j)
                    edge_index[0].append(j)
                    edge_index[1].append(i)  # Add the reverse edge for undirected connection

        return edge_index




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

