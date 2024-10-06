import os
import torch
import random

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
    def __init__(self, stroke_cloud_loops, stroke_node_features, strokes_perpendicular, loop_neighboring_vertical, loop_neighboring_horizontal, loop_neighboring_contained, loop_to_brep, stroke_to_edge):
        super(SketchLoopGraph, self).__init__()

        if stroke_node_features is None: 
            return

        if stroke_to_edge is None or stroke_to_edge.shape[0] == 0:
            stroke_to_edge = np.zeros((stroke_node_features.shape[0], 1))
        
        # Use all 7 values of stroke_node_features + 1 feature of used or not
        self['stroke'].x = torch.cat([torch.tensor(stroke_node_features, dtype=torch.float), 
                                 torch.tensor(stroke_to_edge, dtype=torch.float)], dim=1)


        # Loop node features based on whether the loop is used (repeated 8 times)
        self._compute_loop_features(stroke_cloud_loops, loop_to_brep)

        # Create edges between loops and strokes
        stroke_loop_edges, loop_stroke_edges= self._create_loop_stroke_edges(stroke_cloud_loops)
        self['stroke', 'represents', 'loop'].edge_index = torch.tensor(stroke_loop_edges, dtype=torch.long)
        self['loop', 'represented_by', 'stroke'].edge_index = torch.tensor(loop_stroke_edges, dtype=torch.long)

        # Create neighboring_vertical edges
        self.loop_neighboring_vertical = loop_neighboring_vertical
        vertical_edge_indices = self._create_loop_vertical_neighbor_edges(loop_neighboring_vertical)
        self['loop', 'neighboring_vertical', 'loop'].edge_index = torch.tensor(vertical_edge_indices, dtype=torch.long)

        # Create neighboring_horizontal edges
        horizontal_edge_indices = self._create_loop_neighbor_edges(loop_neighboring_horizontal)
        self['loop', 'neighboring_horizontal', 'loop'].edge_index = torch.tensor(horizontal_edge_indices, dtype=torch.long)

        # Create directed edges for contained loops
        contains_edges, is_contained_by_edges = self._create_containment_edges(loop_neighboring_contained)
        self['loop', 'contains', 'loop'].edge_index = torch.tensor(contains_edges, dtype=torch.long)

        # Create stroke order edges
        stroke_order_edges = self._create_stroke_order_edges(stroke_node_features)
        self['stroke', 'order', 'stroke'].edge_index = torch.tensor(stroke_order_edges, dtype=torch.long)

        # Create stroke connect edges from connected_stroke_nodes
        strokes_perpendicular_edges = self._create_stroke_connect_edges(strokes_perpendicular)
        self['stroke', 'perpendicular', 'stroke'].edge_index = torch.tensor(strokes_perpendicular_edges, dtype=torch.long)


    def to_device(self, device):
        # Target shape (200, 8)
        target_shape = (200, 8)
        
        for node_type in self.node_types:
            if 'x' in self[node_type]:
                x = self[node_type].x
                current_shape = x.shape

                # Check if padding is needed
                if current_shape[0] < target_shape[0]:
                    # Pad to (200, 8) with -1
                    pad_size = (0, target_shape[1] - current_shape[1], 0, target_shape[0] - current_shape[0])  # (pad_last_dim, pad_first_dim, pad_dim_for_nodes)
                    x_padded = torch.nn.functional.pad(x, pad_size, mode='constant', value=-1)
                    self[node_type].x = x_padded
                else:
                    self[node_type].x = x  # No padding needed if already the correct size

                # Move to the specified device
                self[node_type].x = self[node_type].x.to(device)
        
        for edge_type in self.edge_types:
            if 'edge_index' in self[edge_type]:
                self[edge_type].edge_index = self[edge_type].edge_index.to(device)


    def _compute_loop_features(self, stroke_cloud_loops, loop_to_brep):
        """
        Compute loop features. If the loop is used (based on loop_to_brep), assign 1, otherwise 0.
        Repeat the value 8 times to match the feature size of 8.
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

            # Repeat the last_value 8 times for the loop feature
            loop_feature = [last_value] * 8
            loop_features.append(loop_feature)

        self['loop'].x = torch.tensor(loop_features, dtype=torch.float)
        return 

    def set_loop_features(self, loop_to_brep):
        # Erase the current loop features
        num_loops = self['loop'].num_nodes
        self['loop'].x = torch.zeros((num_loops, 8), dtype=torch.float)  # Set all features to 0 initially

        # Recompute the loop features using loop_to_brep
        loop_features = []
        for loop_idx in range(num_loops):
            if loop_to_brep.shape[0] == 0:
                last_value = 0  # If loop_to_brep is empty, set to 0
            else:
                # Compute the last value: 1 if any value in the row is 1, otherwise 0
                last_value = float(loop_to_brep[loop_idx, :].sum() > 0)

            # Repeat the last_value 8 times for the loop feature
            loop_feature = [last_value] * 8
            loop_features.append(loop_feature)

        # Update the loop features
        self['loop'].x = torch.tensor(loop_features, dtype=torch.float)


    def _create_loop_stroke_edges(self, stroke_cloud_loops):
        loop_indices = []
        stroke_indices = []
        
        # Create edges between each loop node and its corresponding stroke nodes
        for loop_idx, loop in enumerate(stroke_cloud_loops):
            for stroke_idx in loop:
                loop_indices.append(loop_idx)  # Connect the current loop node
                stroke_indices.append(stroke_idx)  # To each stroke node in the sublist
        
        return [stroke_indices, loop_indices], [loop_indices, stroke_indices]

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


    def _create_loop_vertical_neighbor_edges(self, loop_neighboring):
        """ Create non-directed edges for neighboring loops """
        loop_edge_indices = ([], [])
        
        num_loops = loop_neighboring.shape[0]
        for i in range(num_loops):
            for j in range(num_loops):
                if loop_neighboring[i, j] != -1:
                    loop_edge_indices[0].append(i)
                    loop_edge_indices[1].append(j)
        
        return loop_edge_indices

    def _create_containment_edges(self, loop_neighboring_contained):
        """
        Create directed containment edges.
        If [i, j] = 1, then loop i contains loop j. Add edges for both 'contains' and 'is_contained_by'.
        
        Returns:
        - contains_edges: Edges for 'contains' relation.
        - is_contained_by_edges: Edges for 'is_contained_by' relation.
        """
        contains_edges = ([], [])
        is_contained_by_edges = ([], [])

        num_loops = loop_neighboring_contained.shape[0]
        for i in range(num_loops):
            for j in range(num_loops):
                if loop_neighboring_contained[i, j] == 1:  # If loop i contains loop j
                    contains_edges[0].append(i)
                    contains_edges[1].append(j)
                    
                    # Reverse the edge for 'is_contained_by'
                    is_contained_by_edges[0].append(j)
                    is_contained_by_edges[1].append(i)

        return contains_edges, is_contained_by_edges

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


    def set_select_sketch(self, sketch_loop_selection_mask):
        """
        Update the loop node features based on the sketch_loop_selection_mask.
        If a loop is selected (value is 1 in sketch_loop_selection_mask), update its features to be 2, repeated 8 times.
        
        Also, for strokes represented by the selected loops, update the last digit of the stroke features to 2.
        
        Parameters:
        sketch_loop_selection_mask (torch.Tensor): A tensor of shape (num_loops, 1), where a value of 1 
                                                indicates that the loop is selected.
        """
        # Ensure the mask is the correct shape
        num_loops = self['loop'].num_nodes
        assert sketch_loop_selection_mask.shape == (num_loops, 1), "Invalid mask shape"

        # Update loop node features based on the mask
        for loop_idx in range(num_loops):
            if sketch_loop_selection_mask[loop_idx].item() == 1:
                # If the loop is selected, set its features to 2, repeated 8 times
                self['loop'].x[loop_idx] = torch.tensor([2] * 8, dtype=torch.float)

        # Get the edges between loops and strokes
        loop_to_stroke_edges = self['loop', 'represented_by', 'stroke'].edge_index

        # Iterate over all loop nodes again to find strokes represented by selected loops
        for loop_idx in range(num_loops):
            if sketch_loop_selection_mask[loop_idx].item() == 1:
                # Find all strokes represented by this loop
                strokes_represented_by_loop = loop_to_stroke_edges[1][loop_to_stroke_edges[0] == loop_idx]
                
                # Update the last digit of the stroke features for the represented strokes
                for stroke_idx in strokes_represented_by_loop:
                    self['stroke'].x[stroke_idx, -1] = 2  # Update the last digit of the stroke feature to 2


    def _full_shape(self):
        """
        Check if all loop nodes in the graph form a full_shape. 
        A loop is considered a full_shape if and only if the sum of its neighboring_vertical and neighboring_horizontal edges (val_1)
        is greater than the number of its representedBy edges to strokes (val_2).

        Returns:
        bool: True if all loops are full shapes, False otherwise.
        """
        # Get the number of loop nodes
        num_loops = self['loop'].num_nodes

        # Get edge indices for each relation
        neighboring_vertical_edges = self['loop', 'neighboring_vertical', 'loop'].edge_index
        neighboring_horizontal_edges = self['loop', 'neighboring_horizontal', 'loop'].edge_index
        represented_by_edges = self['loop', 'represented_by', 'stroke'].edge_index

        # Iterate through all loop nodes
        for loop_idx in range(num_loops):
            # Calculate val_1: sum of neighboring_vertical and neighboring_horizontal edges for this loop
            val_1 = (
                torch.sum(neighboring_vertical_edges[0] == loop_idx).item() +
                torch.sum(neighboring_horizontal_edges[0] == loop_idx).item()
            )
            
            # Calculate val_2: number of representedBy edges for this loop
            val_2 = torch.sum(represented_by_edges[0] == loop_idx).item()

            # Check if it's a full_shape
            if val_1 <= val_2:
                return False  # If any loop is not a full shape, return False

        return True  # All loops are full shapes



    def _has_circle_shape(self):
        """
        Determine if there is a subgraph of loop nodes (filtered by specific conditions) that forms a circular structure
        using 'neighboring_vertical' edges. The subgraph must have > 4 nodes and each loop node must have a vertical
        neighboring edge for all its representedBy stroke indices.

        Returns:
        bool: True if such a full shape exists, False otherwise.
        """
        # Step 1: Filter loop nodes based on the given criteria

        # Get the number of loop nodes
        num_loops = self['loop'].num_nodes

        # Get the 'neighboring_vertical' matrix and representedBy edges
        loop_neighboring_vertical_matrix = self.loop_neighboring_vertical  # np.array
        represented_by_edges = self['loop', 'represented_by', 'stroke'].edge_index

        # Initialize an empty list to store valid loop nodes
        valid_loop_nodes = []

        # Iterate over all loop nodes
        for loop_idx in range(num_loops):
            # Get the loop features (check if the loop is used)
            loop_features = self['loop'].x[loop_idx].tolist()
            
            # Condition A: Skip loops that have features all equal to 1
            if loop_features == [1] * 8:
                continue

            # Condition B: Calculate number of neighboring_vertical and representedBy edges
            num_vertical_edges = (loop_neighboring_vertical_matrix[loop_idx] != -1).sum()
            num_represented_by_edges = (represented_by_edges[0] == loop_idx).sum().item()

            # Check if the number of vertical edges is greater than representedBy edges
            if num_vertical_edges <= num_represented_by_edges:
                continue

            # Now, get the stroke indices for this loop
            stroke_indices = represented_by_edges[1][represented_by_edges[0] == loop_idx].tolist()  # Get the strokes it is connected to
            
            # Get all the stroke indices of neighboring vertical edges from loop_neighboring_vertical_matrix
            neighboring_strokes = set(loop_neighboring_vertical_matrix[loop_idx, loop_neighboring_vertical_matrix[loop_idx] != -1].tolist())

            # Check if all the stroke_indices are present in the neighboring strokes
            if not set(stroke_indices).issubset(neighboring_strokes):
                continue  # Skip this loop if it doesn't satisfy the condition

            # If both conditions are satisfied, add the loop node to valid_loop_nodes
            valid_loop_nodes.append(loop_idx)

        # Step 2: Check if a subgraph formed by neighboring_vertical edges contains a connected component with > 4 nodes

        # If there are fewer than 4 valid loops, return False
        if len(valid_loop_nodes) < 4:
            return False

        # Create a graph using the filtered nodes and the neighboring_vertical edges
        G = nx.Graph()

        # Add valid nodes to the graph
        G.add_nodes_from(valid_loop_nodes)

        # Add neighboring_vertical edges between the valid nodes based on the matrix values
        for i in valid_loop_nodes:
            for j in valid_loop_nodes:
                if loop_neighboring_vertical_matrix[i, j] != -1:  # Check if a valid edge exists
                    G.add_edge(i, j)

        # Step 3: Find if there is a connected component (subgraph) with > 4 nodes
        for component in nx.connected_components(G):
            if len(component) > 4:
                return True  # Found a connected component with more than 4 nodes

        # If no valid subgraph with > 4 nodes is found, return False
        return False




def build_graph(stroke_dict, messy = False):
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




def convert_to_hetero_data(sketch_graph):
    # Create a new HeteroData instance
    hetero_data = HeteroData()

    # Copy over node features ('stroke' and 'loop')
    hetero_data['stroke'].x = sketch_graph['stroke'].x.clone()
    hetero_data['loop'].x = sketch_graph['loop'].x.clone()

    # Copy over edge indices
    hetero_data['stroke', 'represents', 'loop'].edge_index = sketch_graph['stroke', 'represents', 'loop'].edge_index.clone()
    hetero_data['loop', 'represented_by', 'stroke'].edge_index = sketch_graph['loop', 'represented_by', 'stroke'].edge_index.clone()

    # Copy over all other edges
    hetero_data['loop', 'neighboring_vertical', 'loop'].edge_index = sketch_graph['loop', 'neighboring_vertical', 'loop'].edge_index.clone()
    hetero_data['loop', 'neighboring_horizontal', 'loop'].edge_index = sketch_graph['loop', 'neighboring_horizontal', 'loop'].edge_index.clone()
    hetero_data['loop', 'contains', 'loop'].edge_index = sketch_graph['loop', 'contains', 'loop'].edge_index.clone()
    hetero_data['stroke', 'order', 'stroke'].edge_index = sketch_graph['stroke', 'order', 'stroke'].edge_index.clone()
    hetero_data['stroke', 'perpendicular', 'stroke'].edge_index = sketch_graph['stroke', 'perpendicular', 'stroke'].edge_index.clone()

    return hetero_data
