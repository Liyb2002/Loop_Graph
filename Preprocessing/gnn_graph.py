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
        self['stroke'].x = torch.tensor(stroke_loop_embeddings, dtype=torch.float)

        # Converting adjacency matrices to edge indices
        loop_neighboring_vertical_indices = torch.nonzero(torch.tensor(loop_neighboring_vertical.clone().detach(), dtype=torch.long))
        loop_neighboring_horizontal_indices = torch.nonzero(torch.tensor(loop_neighboring_horizontal.clone().detach(), dtype=torch.long))

        # Setting edge indices
        self['stroke', 'verticalNeighboring', 'stroke'].edge_index = loop_neighboring_vertical_indices.t().contiguous()
        self['stroke', 'horizontalNeighboring', 'stroke'].edge_index = loop_neighboring_horizontal_indices.t().contiguous()

        self.build_stroke_loop_representation(stroke_to_brep)
    
    def build_stroke_loop_representation(self, stroke_to_brep):

        if stroke_to_brep.shape[0] == 0:
            is_disconnected = torch.ones(self['stroke'].x.shape[0], 1, dtype=torch.int)
        else:
            is_disconnected = (stroke_to_brep.sum(dim=1) == 0).int().unsqueeze(1)
        
        self['stroke'].x = torch.cat((self['stroke'].x, is_disconnected), dim=1)




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