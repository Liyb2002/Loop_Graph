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
    def __init__(self, stroke_loop, stroke_edges, stroke_cloud_coplanar, stroke_brep_connect):
        super(SketchHeteroData, self).__init__()

        # Node features
        self['stroke'].x = torch.tensor(stroke_loop, dtype=torch.float)

        # Converting adjacency matrices to edge indices
        stroke_edges_indices = torch.nonzero(torch.tensor(stroke_edges.clone().detach(), dtype=torch.long))
        stroke_cloud_coplanar_indices = torch.nonzero(torch.tensor(stroke_cloud_coplanar.clone().detach(), dtype=torch.long))

        # Setting edge indices
        self['stroke', 'strokeIntersect', 'stroke'].edge_index = stroke_edges_indices.t().contiguous()
        self['stroke', 'strokeCoplanar', 'stroke'].edge_index = stroke_cloud_coplanar_indices.t().contiguous()

        self.build_stroke_loop_representation(stroke_brep_connect)
    
    def build_stroke_loop_representation(self, stroke_brep_connect):

        if stroke_brep_connect.shape[1] == 0:
            num_strokes = self['stroke'].x.shape[0]
            is_connected = torch.zeros((num_strokes, 1), device=self['stroke'].x.device)
        else:
            is_connected = (stroke_brep_connect.sum(dim=1) > 0).float().unsqueeze(1)

        self['stroke'].x = torch.cat((self['stroke'].x, is_connected), dim=1)


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

    node_features = np.zeros((num_strokes, 6))
    operations_order_matrix = np.zeros((num_strokes, num_operation_counts+1))


    for i, (_, stroke) in enumerate(stroke_dict.items()):

        # build node_features
        # node_features has shape num_strokes x 6, which is the starting and ending point
        start_point = stroke.vertices[0].position
        end_point = stroke.vertices[1].position
        node_features[i, :3] = start_point
        node_features[i, 3:] = end_point

        # build operation_order_matrix
        # operation_order_matrix has shape num_strokes x num_ops
        for stroke_op_count in stroke.Op_orders:
            operations_order_matrix[i, stroke_op_count] = 1


    return node_features, operations_order_matrix