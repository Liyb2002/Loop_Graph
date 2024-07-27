import os
import torch
import numpy as np

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
    def __init__(self, node_features, operations_matrix, intersection_matrix, operations_order_matrix):
        super(SketchHeteroData, self).__init__()

        # Node features and labels
        self['stroke'].x = node_features
        self['stroke'].y = operations_matrix
        self['stroke'].z = operations_order_matrix
        
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

    def set_brep_connection(self, brep_edge_features, face_feature_gnn_list):
        self['brep'].x = brep_edge_features
        brep_stroke_connection_matrix = self.brep_stroke_cloud_connect(self['stroke'].x, brep_edge_features)
        self.brep_coplanar(face_feature_gnn_list)
        stroke_coplanar_matrix = self.stroke_coplanar()
        return brep_stroke_connection_matrix, stroke_coplanar_matrix
  
    

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

    
    def brep_coplanar(self, face_feature_gnn_list):
        num_nodes = self['brep'].x.shape[0]
        connectivity_matrix = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)

        # Function to find index of an edge in self.brep['x']
        def find_edge_index(edge, brep_edges):
            edge_points = [round(edge[i].item(), 3) for i in range(6)]
            edge_point1 = edge_points[:3]
            edge_point2 = edge_points[3:]

            for idx, brep_edge in enumerate(brep_edges):
                brep_edge_points = [round(brep_edge[i].item(), 3) for i in range(6)]
                brep_edge_point1 = brep_edge_points[:3]
                brep_edge_point2 = brep_edge_points[3:]

                if (edge_point1 == brep_edge_point1 and edge_point2 == brep_edge_point2) or \
                   (edge_point1 == brep_edge_point2 and edge_point2 == brep_edge_point1):
                    return idx
            return -1

        # Iterate over each face
        for face in face_feature_gnn_list:
            face_edge_indices = []
            for edge in face:
                edge_index = find_edge_index(edge, self['brep'].x)
                if edge_index != -1:
                    face_edge_indices.append(edge_index)

            # Mark all edges in the same face as connected
            for i in range(len(face_edge_indices)):
                for j in range(i + 1, len(face_edge_indices)):
                    idx1 = face_edge_indices[i]
                    idx2 = face_edge_indices[j]
                    connectivity_matrix[idx1, idx2] = 1
                    connectivity_matrix[idx2, idx1] = 1

        edge_indices = torch.nonzero(connectivity_matrix == 1).t()
        self['brep', 'brepcoplanar', 'brep'].edge_index = edge_indices.long()


    def stroke_coplanar(self):
        node_features = self['stroke'].x

        x_values = set()
        y_values = set()
        z_values = set()

        # Iterate through each line in node_features
        for line in node_features:
            x1, y1, z1, x2, y2, z2 = line
            x_values.update([x1.item(), x2.item()])
            y_values.update([y1.item(), y2.item()])
            z_values.update([z1.item(), z2.item()])

        # Convert sets to sorted lists
        x_values = sorted(list(x_values))
        y_values = sorted(list(y_values))
        z_values = sorted(list(z_values))

        # Initialize lists to store plane indices
        x_planes = []
        y_planes = []
        z_planes = []

        # Iterate through unique values to create plane lists
        for x in x_values:
            plane = []
            for idx, line in enumerate(node_features):
                x1, y1, z1, x2, y2, z2 = line
                if x1.item() == x and x2.item() == x:
                    plane.append(idx)
            if len(plane) >= 3:
                x_planes.append(plane)

        for y in y_values:
            plane = []
            for idx, line in enumerate(node_features):
                x1, y1, z1, x2, y2, z2 = line
                if y1.item() == y and y2.item() == y:
                    plane.append(idx)
            if len(plane) >= 3:
                y_planes.append(plane)

        for z in z_values:
            plane = []
            for idx, line in enumerate(node_features):
                x1, y1, z1, x2, y2, z2 = line
                if z1.item() == z and z2.item() == z:
                    plane.append(idx)
            if len(plane) >= 3:
                z_planes.append(plane)

        planes = x_planes + y_planes + z_planes

        coplanar_matrix = torch.zeros((node_features.shape[0], node_features.shape[0]), dtype=torch.float32)
        for sublist in planes:
            for i in range(len(sublist)):
                for j in range(i + 1, len(sublist)):
                    coplanar_matrix[sublist[i], sublist[j]] = 1
                    coplanar_matrix[sublist[j], sublist[i]] = 1

        edge_indices = torch.nonzero(coplanar_matrix == 1).t()
        self['stroke', 'strokecoplanar', 'stroke'].edge_index = edge_indices.long()

        return coplanar_matrix




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


def build_loop_connectivity(face_to_stroke):
    num_faces = len(face_to_stroke)
    connectivity_matrix = np.zeros((num_faces, num_faces), dtype=np.float32)

    for i in range(num_faces):
        for j in range(i, num_faces):
            if i == j:
                connectivity_matrix[i, j] = 1  # Each face is connected to itself
            else:
                if len(set(face_to_stroke[i]).intersection(set(face_to_stroke[j]))) > 0:
                    connectivity_matrix[i, j] = 1
                    connectivity_matrix[j, i] = 1  # Ensure symmetry
    
    return connectivity_matrix



