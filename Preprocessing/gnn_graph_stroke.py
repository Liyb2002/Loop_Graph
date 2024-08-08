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
    def __init__(self, node_features, intersection_matrix):
        super(SketchHeteroData, self).__init__()

        # Node features and labels
        self['stroke'].x = node_features
        
        # Order of nodes (sequential order of reading)
        num_strokes = node_features.shape[0]
        order = list(range(num_strokes))
        self['stroke'].order = torch.tensor(order, dtype=torch.long)
        
        # Intersection matrix to edge indices
        intersection_matrix = intersection_matrix.squeeze(0)
        edge_indices = torch.nonzero(intersection_matrix == 1).t()
        self['stroke', 'intersects', 'stroke'].edge_index = edge_indices.long()

        # Temporal edge index (order of nodes)
        temporal_edge_index = [order[:-1], order[1:]]
        temporal_edge_tensor = torch.tensor(temporal_edge_index, dtype=torch.long).contiguous()
        self['stroke', 'temp_previous', 'stroke'].edge_index = temporal_edge_tensor


        self.intersection_matrix = intersection_matrix

    def set_brep_connection(self, brep_edge_features):
        self['brep'].x = brep_edge_features
        brep_stroke_connection_matrix = self.brep_stroke_cloud_connect(self['stroke'].x, brep_edge_features)
        return brep_stroke_connection_matrix
  
    

    def brep_stroke_cloud_connect(self, node_features, edge_features):
        n = node_features.shape[0]
        m = edge_features.shape[0]
        
        # Initialize the (n, m) matrix with zeros
        connection_matrix = torch.zeros((n, m), dtype=torch.float32)
        
        # Compare each node feature with each edge feature
        for i in range(n):
            for j in range(m):
                if torch.equal(node_features[i], edge_features[j]):
                    connection_matrix[i, j] = 1
        
        edge_indices = torch.nonzero(connection_matrix == 1).t()
        self['stroke', 'represented_by', 'brep'].edge_index = edge_indices.long()

        return connection_matrix

    


def build_graph(stroke_dict):
    num_strokes = len(stroke_dict)
    num_operations = len(operations_dict)
    num_operation_counts = 0

    # find the total number of operations
    for i, (_, stroke) in enumerate(stroke_dict.items()):
        for index in stroke.Op_orders:
            if index > num_operation_counts:
                num_operation_counts = index

    # a map that maps stroke_id (e.g 'edge_0_0' to 0)
    stroke_id_to_index = {}



    node_features = np.zeros((num_strokes, 6))
    operations_matrix = np.zeros((num_strokes, num_operations))
    intersection_matrix = np.zeros((num_strokes, num_strokes))
    operations_order_matrix = np.zeros((num_strokes, num_operation_counts+1))


    for i, (_, stroke) in enumerate(stroke_dict.items()):

        # build node_features
        # node_features has shape num_strokes x 6, which is the starting and ending point
        start_point = stroke.vertices[0].position
        end_point = stroke.vertices[1].position
        node_features[i, :3] = start_point
        node_features[i, 3:] = end_point

        # build operations_matrix
        # operations_matrix has shape num_strokes x num_type_ops
        for op in stroke.Op:
            if op in operations_dict:
                op_index = operations_dict[op]
                operations_matrix[i, op_index] = 1
        
        stroke_id_to_index = {stroke_id: index for index, stroke_id in enumerate(stroke_dict)}

        # build intersection_matrix
        # intersection_matrix has shape num_strokes x num_strokes
        for connected_id in stroke.connected_edges:
            if connected_id in stroke_id_to_index:
                j = stroke_id_to_index[connected_id]
                intersection_matrix[i, j] = 1
                intersection_matrix[j, i] = 1

        # build operation_order_matrix
        # operation_order_matrix has shape num_strokes x num_ops
        for stroke_op_count in stroke.Op_orders:
            operations_order_matrix[i, stroke_op_count] = 1


    # graph = SketchHeteroData(node_features, operations_matrix, intersection_matrix)
    # graph.output_info()

    return node_features, operations_matrix, intersection_matrix, operations_order_matrix
