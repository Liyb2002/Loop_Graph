import numpy as np
import random
from shapely.geometry import Polygon, Point
from shapely.geometry.polygon import orient
from shapely import affinity
import pyrr
import json 
import torch

import matplotlib.pyplot as plt
from itertools import permutations, combinations
import networkx as nx
from concurrent.futures import ThreadPoolExecutor, as_completed


def compute_normal(face_vertices, other_point):
    if len(face_vertices) < 3:
        raise ValueError("Need at least three points to define a plane")


    p1 = np.array(face_vertices[0].position)
    p2 = np.array(face_vertices[1].position)
    p3 = np.array(face_vertices[2].position)

    # Create vectors from the first three points
    v1 = p2 - p1
    v2 = p3 - p1

    # Compute the cross product to find the normal
    normal = np.cross(v1, v2)

    norm = np.linalg.norm(normal)
    if norm == 0:
        raise ValueError("The points do not form a valid plane")
    normal_unit = normal / norm

    # Use the other point to check if the normal should be flipped
    reference_vector = other_point.position - p1
    if np.dot(normal_unit, reference_vector) > 0:
        normal_unit = -normal_unit  # Flip the normal if it points towards the other point

    return normal_unit.tolist()


#----------------------------------------------------------------------------------#


def round_position(position, decimals=3):
    return tuple(round(coord, decimals) for coord in position)



#----------------------------------------------------------------------------------#




def find_target_verts(target_vertices, edges) :
    target_pos_1 = round_position(target_vertices[0])
    target_pos_2 = round_position(target_vertices[1])
    target_positions = {target_pos_1, target_pos_2}
    
    for edge in edges:
        verts = edge.vertices()
        if len(verts) ==2 :
            edge_positions = {
                round_position([verts[0].X, verts[0].Y, verts[0].Z]), 
                round_position([verts[1].X, verts[1].Y, verts[1].Z])
                }
        
            if edge_positions == target_positions:
                return edge
        
    return None


#----------------------------------------------------------------------------------#


def get_neighbor_verts(vert, non_app_edge, Edges):
    #get the neighbor of the given vert
    neighbors = []
    for edge in Edges:
        if edge.id == non_app_edge.id:
            continue
        if edge.vertices[0].id == vert.id:
            neighbors.append(edge.vertices[1])
        elif edge.vertices[1].id == vert.id:
            neighbors.append(edge.vertices[0])  

    return neighbors

def find_edge_from_verts(vert_1, vert_2, edges):
    vert_1_id = vert_1.id  # Get the ID of vert_1
    vert_2_id = vert_2.id  # Get the ID of vert_2

    for edge in edges:
        # Get the IDs of the vertices in the current edge
        edge_vertex_ids = [vertex.id for vertex in edge.vertices]

        # Check if both vertex IDs are present in the current edge
        if vert_1_id in edge_vertex_ids and vert_2_id in edge_vertex_ids:
            return edge  # Return the edge that contains both vertices

    return None  # Return None if no edge contains both vertices
    

#----------------------------------------------------------------------------------#

def compute_fillet_new_vert(old_vert, neighbor_verts, amount):
    #given the old_vertex (chosen by fillet op), and the neighbor verts, compute the position to move them
    move_positions = []
    old_position = old_vert.position
    
    for neighbor_vert in neighbor_verts:
        direction_vector = [neighbor_vert.position[i] - old_position[i] for i in range(len(old_position))]
        
        norm = sum(x**2 for x in direction_vector)**0.5
        normalized_vector = [x / norm for x in direction_vector]
        
        move_position = [old_position[i] + normalized_vector[i] * amount for i in range(len(old_position))]
        move_positions.append(move_position)
    
    return move_positions


#----------------------------------------------------------------------------------#

def find_rectangle_on_plane(points, normal):
    """
    Find a new rectangle on the same plane as the given larger rectangle, with a translation.
    
    Args:
        points: List of 4 numpy arrays representing the vertices of the larger rectangle.
    
    Returns:
        list: A list of 4 numpy arrays representing the vertices of the new rectangle.
    """
    # Convert points to numpy array for easy manipulation
    points = np.array(points)
    
    # Extract the coordinates
    x_vals = points[:, 0]
    y_vals = points[:, 1]
    z_vals = points[:, 2]
    
    # Check which coordinate is the same for all points (defining the plane)
    if np.all(x_vals == x_vals[0]):
        fixed_coord = 'x'
        fixed_value = x_vals[0]
    elif np.all(y_vals == y_vals[0]):
        fixed_coord = 'y'
        fixed_value = y_vals[0]
    elif np.all(z_vals == z_vals[0]):
        fixed_coord = 'z'
        fixed_value = z_vals[0]
    
    # Determine the min and max for the other two coordinates
    if fixed_coord == 'x':
        min_y, max_y = np.min(y_vals), np.max(y_vals)
        min_z, max_z = np.min(z_vals), np.max(z_vals)
        new_min_y = min_y + (max_y - min_y) * 0.1
        new_max_y = max_y - (max_y - min_y) * 0.1
        new_min_z = min_z + (max_z - min_z) * 0.1
        new_max_z = max_z - (max_z - min_z) * 0.1
        new_points = [
            np.array([fixed_value, new_min_y, new_min_z]),
            np.array([fixed_value, new_max_y, new_min_z]),
            np.array([fixed_value, new_max_y, new_max_z]),
            np.array([fixed_value, new_min_y, new_max_z])
        ]
    elif fixed_coord == 'y':
        min_x, max_x = np.min(x_vals), np.max(x_vals)
        min_z, max_z = np.min(z_vals), np.max(z_vals)
        new_min_x = min_x + (max_x - min_x) * 0.1
        new_max_x = max_x - (max_x - min_x) * 0.1
        new_min_z = min_z + (max_z - min_z) * 0.1
        new_max_z = max_z - (max_z - min_z) * 0.1
        new_points = [
            np.array([new_min_x, fixed_value, new_min_z]),
            np.array([new_max_x, fixed_value, new_min_z]),
            np.array([new_max_x, fixed_value, new_max_z]),
            np.array([new_min_x, fixed_value, new_max_z])
        ]
    elif fixed_coord == 'z':
        min_x, max_x = np.min(x_vals), np.max(x_vals)
        min_y, max_y = np.min(y_vals), np.max(y_vals)
        new_min_x = min_x + (max_x - min_x) * 0.1
        new_max_x = max_x - (max_x - min_x) * 0.1
        new_min_y = min_y + (max_y - min_y) * 0.1
        new_max_y = max_y - (max_y - min_y) * 0.1
        new_points = [
            np.array([new_min_x, new_min_y, fixed_value]),
            np.array([new_max_x, new_min_y, fixed_value]),
            np.array([new_max_x, new_max_y, fixed_value]),
            np.array([new_min_x, new_max_y, fixed_value])
        ]
    
    return new_points


def find_triangle_on_plane(points, normal):

    four_pts = find_rectangle_on_plane(points, normal)
    idx1, idx2 = 0, 1
    point1 = four_pts[idx1]
    point2 = four_pts[idx2]

    point3 = 0.5 * (four_pts[2] + four_pts[3])

    return [point1, point2, point3]


def find_triangle_to_cut(points, normal):

    points = np.array(points)
    
    # Randomly shuffle the indices to choose three points
    start_index = np.random.randint(0, 4)

    # Determine the indices of the three points
    indices = [(start_index + i) % 4 for i in range(3)]

    
    # Use the second point as the pin point
    pin_index = indices[1]
    pin_point = points[pin_index]
    
    # Interpolate between the pin point and the other two points
    point1 = 0.5 * (pin_point + points[indices[0]])
    point2 = 0.5 * (pin_point + points[indices[2]])

    return [pin_point, point1, point2]


def random_circle(points, normal):
    four_pts = find_rectangle_on_plane(points, normal)

    pt = random.choice(four_pts)

    return pt




#----------------------------------------------------------------------------------#




def project_points(feature_lines, obj_center, img_dims=[1000, 1000]):

    obj_center = np.array(obj_center)
    cam_pos = obj_center + np.array([5,0,5])
    up_vec = np.array([0,1,0])
    view_mat = pyrr.matrix44.create_look_at(cam_pos,
                                            np.array([0, 0, 0]),
                                            up_vec)
    near = 0.001
    far = 1.0
    total_view_points = []

    for edge_info in feature_lines:
        view_points = []
        vertices = edge_info['vertices']
        if edge_info['is_curve']:
            vertices = edge_info['sampled_points']
        for p in vertices:
            p -= obj_center
            hom_p = np.ones(4)
            hom_p[:3] = p
            proj_p = np.matmul(view_mat.T, hom_p)
            view_points.append(proj_p)
            
            total_view_points.append(proj_p)
        edge_info['projected_edge'].append(np.array(view_points))
    
    # for edge_info in feature_lines:
    #    plt.plot(edge_info['projected_edge'][0][:, 0], edge_info['projected_edge'][0][:, 1], c="black")
    # plt.show()



    total_view_points = np.array(total_view_points)
    max = np.array([np.max(total_view_points[:, 0]), np.max(total_view_points[:, 1]), np.max(total_view_points[:, 2])])
    min = np.array([np.min(total_view_points[:, 0]), np.min(total_view_points[:, 1]), np.min(total_view_points[:, 2])])

    max_dim = np.maximum(np.abs(max[0]-min[0]), np.abs(max[1]-min[1]))
    proj_mat = pyrr.matrix44.create_perspective_projection_matrix_from_bounds(left=-max_dim/2, right=max_dim/2, bottom=-max_dim/2, top=max_dim/2,
                                                                              near=near, far=far)

    total_projected_points = []
    projected_edges = []

    for edge_info in feature_lines:
        projected_points = []
        for p in edge_info['projected_edge'][0]:
            proj_p = np.matmul(proj_mat, p)
            proj_p[:3] /= proj_p[-1]
            total_projected_points.append(proj_p[:2])
            projected_points.append(proj_p[:2])
        projected_edges.append(np.array(projected_points))

        edge_info['projected_edge'] = projected_edges[-1]
    total_projected_points = np.array(total_projected_points)

    # screen-space
    # scale to take up 80% of the image
    max = np.array([np.max(total_projected_points[:, 0]), np.max(total_projected_points[:, 1])])
    min = np.array([np.min(total_projected_points[:, 0]), np.min(total_projected_points[:, 1])])
    bbox_diag = np.linalg.norm(max - min)
    screen_diag = np.sqrt(img_dims[0]*img_dims[0]+img_dims[1]*img_dims[1])


    for edge_info in feature_lines:
        scaled_points = []
        for p in edge_info['projected_edge']:
            p[1] *= -1
            p *= 0.5*screen_diag/bbox_diag
            p += np.array([img_dims[0]/2, img_dims[1]/2])
            scaled_points.append(p)
        edge_info['projected_edge'] = np.array(scaled_points)

    
    # for edge_info in feature_lines:
    #     f_line = edge_info['projected_edge']
    #     plt.plot(f_line[:, 0], f_line[:, 1], c="black")
    # plt.show()

    return feature_lines


#----------------------------------------------------------------------------------#

def program_to_string(file_path):

    Op_string = []
    with open(file_path, 'r') as file:
        data = json.load(file)
        for Op in data:
            Op_string.append(Op['operation'][0])

    return Op_string


def program_to_tensor(program):
    operation_to_index = {'terminate': 0, 'sketch': 1, 'extrude': 2, 'fillet': 3}
    Op_indices = []

    for Op in program:
        Op_indices.append(operation_to_index[Op])

    return torch.tensor(Op_indices, dtype=torch.long)


def sketch_face_selection(file_path):

    boundary_points = []
    with open(file_path, 'r') as file:
        data = json.load(file)
        for Op in data:
            if Op['operation'][0] == 'sketch':
                boundary_points.append(Op['operation'][1])
            else:
                boundary_points.append([])

    return boundary_points

#----------------------------------------------------------------------------------#

def expected_extrude_point(point, sketch_face_normal, extrude_amount):
    x, y, z = point
    a, b, c = sketch_face_normal
    x_extruded = x - a * extrude_amount
    y_extruded = y - b * extrude_amount
    z_extruded = z - c * extrude_amount
    return [x_extruded, y_extruded, z_extruded]

def canvas_has_point(canvas, point):
    edges = canvas.edges()    
    point = round_position(point)
    
    for edge in edges:
        verts = edge.vertices()
        if len(verts) ==2 :
            edge_positions = [
                round_position([verts[0].X, verts[0].Y, verts[0].Z]), 
                round_position([verts[1].X, verts[1].Y, verts[1].Z])
                ]
    
            if point == edge_positions[0] or point == edge_positions[1]:
                return True
        
    return False

def print_canvas_points(canvas):
    edges = canvas.edges()    
    
    for edge in edges:
        verts = edge.vertices()
        if len(verts) ==2 :
            edge_positions = [
                round_position([verts[0].X, verts[0].Y, verts[0].Z]), 
                round_position([verts[1].X, verts[1].Y, verts[1].Z])
                ]
        print("edge_positions", edge_positions)




#----------------------------------------------------------------------------------#



def preprocess_features(features):
    processed_features = [] 
    for _, f in features:
        processed_features.append(f)
    
    return torch.tensor(processed_features)



#----------------------------------------------------------------------------------#


def face_to_stroke(stroke_cloud_faces, stroke_features):
    num_strokes = stroke_features.shape[0]
    stroke_ids_per_face = []
    
    for face_id, face in stroke_cloud_faces.items():
        face_stroke_ids = []
        # Get all combinations of two vertices
        vertex_combinations = list(combinations(face.vertices, 2))
        
        for comb in vertex_combinations:
            vert1_pos = np.array(comb[0].position)
            vert2_pos = np.array(comb[1].position)
            
            for stroke_id in range(num_strokes):
                start_point = stroke_features[stroke_id, :3]
                end_point = stroke_features[stroke_id, 3:]
                
                if (np.allclose(vert1_pos, start_point) and np.allclose(vert2_pos, end_point)) or \
                   (np.allclose(vert1_pos, end_point) and np.allclose(vert2_pos, start_point)):
                    face_stroke_ids.append(stroke_id)
                    break
        
        stroke_ids_per_face.append(face_stroke_ids)
    
    return stroke_ids_per_face



#----------------------------------------------------------------------------------#


def chosen_face_id(boundary_points, edge_features):
    print("edge_features", len(edge_features))
    print("boundary_points", len(boundary_points))

    

#----------------------------------------------------------------------------------#
def face_aggregate_networkx(stroke_matrix):
    """
    This function finds all valid loops of strokes with size 3 or 4 using NetworkX.
    It generates all possible valid cycles by permuting over nodes and checking for a valid cycle.

    Parameters:
    stroke_matrix (numpy.ndarray): A matrix of shape (num_strokes, 7) where each row represents a stroke
                                   with start and end points in 3D space.

    Returns:
    list: A list of indices of valid loops of strokes, where each loop contains either 3 or 4 strokes.
    """
    
    # Ensure input is a numpy array and ignore the last column
    stroke_matrix = np.array(stroke_matrix)[:, :6]
    
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

    # Generate all possible combinations of nodes of size 3 or 4
    nodes = list(G.nodes)

    # Helper function to check if a set of edges forms a valid cycle
    def check_valid_edges(edges):
        point_count = {}
        for edge in edges:
            point_count[edge[0]] = point_count.get(edge[0], 0) + 1
            point_count[edge[1]] = point_count.get(edge[1], 0) + 1
        # A valid cycle has each node exactly twice
        return all(count == 2 for count in point_count.values())

    # Check for valid loops of size 3 and 4
    for group_nodes in combinations(nodes, 3):
        # Check if these nodes can form a valid subgraph
        if nx.is_connected(G.subgraph(group_nodes)):
            # Generate all permutations of the edges
            for perm_edges in permutations(combinations(group_nodes, 2), 3):
                if check_valid_edges(perm_edges):
                    strokes_in_group = [edge_to_stroke_id.get(edge) or edge_to_stroke_id.get((edge[1], edge[0])) for edge in perm_edges]
                    if None not in strokes_in_group:  # Ensure all edges are found in the mapping
                        valid_groups.append(sorted(strokes_in_group))

    for group_nodes in combinations(nodes, 4):
        # Check if these nodes can form a valid subgraph
        if nx.is_connected(G.subgraph(group_nodes)):
            # Generate all permutations of the edges
            for perm_edges in permutations(combinations(group_nodes, 2), 4):
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


def face_aggregate_direct(stroke_matrix):
    """
    This function finds all connected groups of strokes with size 3 or 4
    by directly checking if each point appears exactly twice within each group.

    Parameters:
    stroke_matrix (numpy.ndarray): A matrix of shape (num_strokes, 7) where each row represents a stroke
                                   with start and end points in 3D space.

    Returns:
    list: A list of indices of valid connected groups of strokes, where each group contains either 3 or 4 strokes.
    """
    
    # Ensure input is a numpy array and ignore the last column
    stroke_matrix = np.array(stroke_matrix)[:, :6]
    
    # Get the number of strokes
    num_strokes = stroke_matrix.shape[0]
    
    # List to store valid groups
    valid_groups = []
    
    # Function to check if all points appear exactly twice
    def check_group(group_indices):
        point_count = {}
        for idx in group_indices:
            start_point = tuple(np.round(stroke_matrix[idx, :3], 4))
            end_point = tuple(np.round(stroke_matrix[idx, 3:], 4))
            point_count[start_point] = point_count.get(start_point, 0) + 1
            point_count[end_point] = point_count.get(end_point, 0) + 1
        # Check if all points appear exactly twice
        return all(count == 2 for count in point_count.values())
    
    # Check all combinations of 3 strokes
    for group_indices in combinations(range(num_strokes), 3):
        if check_group(group_indices):
            valid_groups.append(list(group_indices))
    
    # Check all combinations of 4 strokes
    for group_indices in combinations(range(num_strokes), 4):
        if check_group(group_indices):
            valid_groups.append(list(group_indices))
    
    return valid_groups



#----------------------------------------------------------------------------------#
def reorder_loops(loops):
    """
    Reorder loops based on the average order of their strokes.
    
    Parameters:
    loops (list of list of int): A list where each sublist contains indices representing strokes of a loop.
    
    Returns:
    list of list of int: The reordered loops where the smallest loop order comes first.
    """
    # Calculate the order for each loop (average of stroke indices)
    loop_orders = [(i, sum(loop) / len(loop)) for i, loop in enumerate(loops)]

    # Sort loops by their calculated order
    loop_orders.sort(key=lambda x: x[1])

    # Reorder loops according to the sorted order
    reordered_loops = [loops[i] for i, _ in loop_orders]

    return reordered_loops


def swap_rows_with_probability(matrix_a, matrix_b, swap_range=5, swap_prob=0.3):
    num_rows = matrix_a.shape[0]
    
    for i in range(num_rows):
        # Perform the swap with the given probability
        if random.random() < swap_prob:
            # Determine the range of valid indices to swap with
            lower_bound = max(0, i - swap_range)
            upper_bound = min(num_rows - 1, i + swap_range)
            
            # Choose a random index within the swap range
            swap_idx = random.randint(lower_bound, upper_bound)
            
            # Swap rows in both matrices if the chosen index is different from the current index
            if swap_idx != i:
                matrix_a[[i, swap_idx]] = matrix_a[[swap_idx, i]]
                matrix_b[[i, swap_idx]] = matrix_b[[swap_idx, i]]
    
    return matrix_a, matrix_b


#----------------------------------------------------------------------------------#
def loop_neighboring_simple(loops):
    """
    Determine neighboring loops based on shared edges.
    
    Parameters:
    loops (list of list of int): A list where each sublist contains indices representing edges of a loop.
    
    Returns:
    np.ndarray: A matrix of shape (num_loops, num_loops) where [i, j] is 1 if loops i and j share an edge, otherwise 0.
    """
    num_loops = len(loops)
    # Initialize the neighboring matrix with zeros with dtype float32
    neighboring_matrix = np.zeros((num_loops, num_loops), dtype=np.float32)
    
    # Iterate over each pair of loops to check for shared edges
    for i in range(num_loops):
        for j in range(i + 1, num_loops):
            # Check if loops i and j share any edge
            if set(loops[i]).intersection(set(loops[j])):
                neighboring_matrix[i, j] = 1.0
                neighboring_matrix[j, i] = 1.0  # Since the matrix is symmetric
    
    return neighboring_matrix
    


def loop_neighboring_complex(loops, stroke_node_features):
    """
    Determine neighboring loops based on shared edges and different normals.
    
    Parameters:
    loops (list of list of int): A list where each sublist contains indices representing edges of a loop.
    stroke_node_features (np.ndarray): A (num_strokes, 7) matrix where the first 6 columns represent two 3D points forming a line.
    
    Returns:
    np.ndarray: A matrix of shape (num_loops, num_loops) where [i, j] is 1 if loops i and j are connected, otherwise 0.
    """
    num_loops = len(loops)
    # Initialize the neighboring matrix with zeros with dtype float32
    neighboring_matrix = np.zeros((num_loops, num_loops), dtype=np.float32)
    
    # Function to compute the normal of a loop
    def compute_normal(loop_indices):
        # List to store edges, each with 6 useful values
        edges = []
        
        # Extract edges from loop_indices and discard the last value
        for idx in loop_indices:
            edge = stroke_node_features[idx, :6]  # First 6 values represent two 3D points
            edges.append(edge)
        
        # Find two edges that share a common point
        shared_point_found = False
        for i in range(len(edges)):
            for j in range(i + 1, len(edges)):
                # Extract points from edges
                p1_start, p1_end = edges[i][:3], edges[i][3:6]
                p2_start, p2_end = edges[j][:3], edges[j][3:6]

                # Check if there is a common point between these two edges
                if np.allclose(p1_start, p2_start) or np.allclose(p1_start, p2_end):
                    common_point = p1_start
                    vec1 = p1_end - p1_start
                    vec2 = p2_end - p2_start if np.allclose(p1_start, p2_start) else p2_start - p2_end
                    shared_point_found = True
                    break
                elif np.allclose(p1_end, p2_start) or np.allclose(p1_end, p2_end):
                    common_point = p1_end
                    vec1 = p1_start - p1_end
                    vec2 = p2_end - p2_start if np.allclose(p1_end, p2_start) else p2_start - p2_end
                    shared_point_found = True
                    break
            if shared_point_found:
                break
        
        # Compute the cross product to get the normal
        normal = np.cross(vec1, vec2)
        
        if np.isnan(normal).any():
            return np.array([0,0,0])
        normal = normal / np.linalg.norm(normal)
        
        # Ensure the normal is positive in the z-direction
        if normal[2] < 0:
            normal = -normal

        return normal

    # Compute normals for each loop
    loop_normals = [compute_normal(loop) for loop in loops]

    # Iterate over each pair of loops to check for shared edges and different normals
    for i in range(num_loops):
        for j in range(i + 1, num_loops):
            # Check if loops i and j share any edge
            if set(loops[i]).intersection(set(loops[j])):
                # Check if the normals are different
                if not np.allclose(loop_normals[i], loop_normals[j]):
                    neighboring_matrix[i, j] = 1.0
                    neighboring_matrix[j, i] = 1.0  # Since the matrix is symmetric

    return neighboring_matrix



def coplanr_neighorbing_loop(matrix1, matrix2):
    """
    Compute a matrix indicating coplanar neighboring loops based on the differences between matrix1 and matrix2.
    
    Parameters:
    matrix1 (np.ndarray): A matrix of shape (num_loops, num_loops) with more values of 1 than matrix2.
    matrix2 (np.ndarray): A matrix of shape (num_loops, num_loops) with fewer values of 1 than matrix1.
    
    Returns:
    np.ndarray: A matrix of the same shape where [i, j] is 1 if matrix1[i, j] is 1 and matrix2[i, j] is not 1.
    """
    # Ensure matrices have the same shape
    if matrix1.shape != matrix2.shape:
        raise ValueError("Both matrices must have the same shape.")
    
    # Compute the coplanar neighboring matrix
    result_matrix = np.where((matrix1 == 1) & (matrix2 != 1), 1, 0)
    
    return result_matrix
    


def loop_contained(loops, stroke_node_features):
    """
    Determine if a loop is contained within another loop based on bounding boxes in 3D space.
    
    Parameters:
    loops (list of list of int): A list where each sublist contains indices representing strokes of a loop.
    stroke_node_features (dict): A dictionary where the key is the stroke index, and the value is a list of two 3D points [(x1, y1, z1), (x2, y2, z2)] representing the stroke.
    
    Returns:
    np.ndarray: A matrix of shape (num_loops, num_loops) where [i, j] is 1 if loop i contains loop j, otherwise 0.
    """
    num_loops = len(loops)
    
    # Initialize the contained matrix with zeros
    contained_matrix = np.zeros((num_loops, num_loops), dtype=np.float32)
    
    # Step 1: Calculate the bounding box (min_x, max_x, min_y, max_y, min_z, max_z) for each loop
    bounding_boxes = []
    
    for loop in loops:
        # Initialize min/max values with extreme values
        min_x, max_x = float('inf'), float('-inf')
        min_y, max_y = float('inf'), float('-inf')
        min_z, max_z = float('inf'), float('-inf')
        
        # Process each stroke in the loop
        for stroke in loop:
            stroke_coords = stroke_node_features[stroke]  # Each stroke has exactly 6 values: [x1, y1, z1, x2, y2, z2]
            x1, y1, z1, x2, y2, z2, _ = stroke_coords
            
            # Update bounding box for the loop
            min_x, max_x = min(min_x, x1, x2), max(max_x, x1, x2)
            min_y, max_y = min(min_y, y1, y2), max(max_y, y1, y2)
            min_z, max_z = min(min_z, z1, z2), max(max_z, z1, z2)
        
        # Store the bounding box as a tuple (min_x, max_x, min_y, max_y, min_z, max_z)
        bounding_boxes.append((min_x, max_x, min_y, max_y, min_z, max_z))
    
    # Step 2: Check if one loop is contained in another in 3D space
    for i in range(num_loops):
        for j in range(num_loops):
            if i != j:  # Avoid comparing the same loop
                min_x_i, max_x_i, min_y_i, max_y_i, min_z_i, max_z_i = bounding_boxes[i]
                min_x_j, max_x_j, min_y_j, max_y_j, min_z_j, max_z_j = bounding_boxes[j]
                
                # Check containment conditions in 3D space
                if (min_x_i <= min_x_j <= max_x_j <= max_x_i) and \
                   (min_y_i <= min_y_j <= max_y_j <= max_y_i) and \
                   (min_z_i <= min_z_j <= max_z_j <= max_z_i):
                    contained_matrix[i, j] = 1.0  # Loop i contains loop j
    
    return contained_matrix



def loop_coplanar(loops, stroke_node_features):
    """
    Determine which loops are co-planar by comparing common values in the x, y, or z axes.
    
    Parameters:
    loops (list of lists): A list where each sublist contains indices of strokes forming a loop.
    stroke_node_features (np.ndarray): A NumPy array of shape (num_strokes, 7), where the first 6 values represent two 3D points.

    Returns:
    coplanar_matrix (np.ndarray): A binary matrix of shape (num_loops, num_loops), where [i, j] = 1 if loops i and j are co-planar.
    """
    num_loops = len(loops)
    coplanar_matrix = np.zeros((num_loops, num_loops), dtype=np.float32)

    # To store the (axis, value) for each loop
    loop_features = []

    # Step 1: Compute the (axis, value) for each loop
    for loop in loops:
        points = []
        # Gather all the points from the strokes in the loop
        for stroke_idx in loop:
            stroke_points = stroke_node_features[stroke_idx, :6]  # First 6 values represent two 3D points
            points.append(stroke_points[:3])  # Start point
            points.append(stroke_points[3:6])  # End point
        
        points = np.array(points)
        
        # Check if all points share the same value in the x, y, or z axis
        if np.all(points[:, 0] == points[0, 0]):
            loop_features.append(('x', points[0, 0]))
        elif np.all(points[:, 1] == points[0, 1]):
            loop_features.append(('y', points[0, 1]))
        elif np.all(points[:, 2] == points[0, 2]):
            loop_features.append(('z', points[0, 2]))
        else:
            loop_features.append(('none', 0))

    # Step 2: Determine co-planar loops
    for i in range(num_loops):
        for j in range(i + 1, num_loops):
            # If two loops have the same (axis, value), they are co-planar
            if loop_features[i] == loop_features[j]:
                coplanar_matrix[i, j] = 1
                coplanar_matrix[j, i] = 1  # Symmetric for undirected relationship

    return coplanar_matrix
    

def check_validacy(matrix1, matrix2):
    """
    Check the validity between two matrices.
    
    Parameters:
    matrix1 (np.ndarray): The first matrix of shape (num_loops, num_loops).
    matrix2 (np.ndarray): The second matrix of shape (num_loops, num_loops).
    
    Returns:
    bool: True if for every entry [i, j] in matrix2 with value 1, the corresponding entry in matrix1 is also 1.
    """
    # Ensure matrices have the same shape
    if matrix1.shape != matrix2.shape:
        raise ValueError("Both matrices must have the same shape.")
    
    # Check validity
    is_valid = np.all((matrix2 == 1) <= (matrix1 == 1))
    
    return is_valid



def connected_strokes(stroke_node_features):
    """
    Returns a matrix indicating if two strokes share a common point.
    
    Parameters:
    stroke_node_features (np.ndarray): A numpy array of shape (num_strokes, 7), where the first 6 values
                                       represent two 3D points of a stroke and the last value is unused.
    
    Returns:
    connected (np.ndarray): A binary matrix of shape (num_strokes, num_strokes), where [i, j] is 1 if stroke i and stroke j share a common point.
    """
    num_strokes = stroke_node_features.shape[0]
    
    # Initialize the connected matrix with zeros
    connected = np.zeros((num_strokes, num_strokes), dtype=np.float32)
    
    # Iterate over each pair of strokes
    for i in range(num_strokes):
        stroke_i_start = stroke_node_features[i, :3]  # First point of stroke i
        stroke_i_end = stroke_node_features[i, 3:6]   # Second point of stroke i
        
        for j in range(i + 1, num_strokes):
            stroke_j_start = stroke_node_features[j, :3]  # First point of stroke j
            stroke_j_end = stroke_node_features[j, 3:6]   # Second point of stroke j

            # Check if stroke i and stroke j share a common point (either order)
            if (np.allclose(stroke_i_start, stroke_j_start) or np.allclose(stroke_i_start, stroke_j_end) or
                np.allclose(stroke_i_end, stroke_j_start) or np.allclose(stroke_i_end, stroke_j_end)):
                
                connected[i, j] = 1
                connected[j, i] = 1  # Symmetric connection since it's undirected

    return connected
 

#----------------------------------------------------------------------------------#
def stroke_to_brep(stroke_cloud_loops, brep_loops, stroke_node_features, final_brep_edges):
    """
    Find the correspondence between stroke loops and brep loops and visualize unrepresented stroke loops.
    
    Parameters:
    stroke_cloud_loops (list of list of int): Each sublist contains indices of strokes in stroke_node_features.
    brep_loops (list of list of int): Each sublist contains indices of edges in final_brep_edges.
    stroke_node_features (np.ndarray): A matrix of shape (num_strokes, 7), where the first 6 columns represent two 3D points.
    final_brep_edges (np.ndarray): A matrix of shape (num_brep_edges, 6) representing two 3D points for each edge.
    
    Returns:
    np.ndarray: A matrix with shape (num_stroke_cloud_loops, num_brep_loops) where each entry is 1 if the loops match, otherwise 0.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    # Initialize the correspondence matrix
    num_stroke_cloud_loops = len(stroke_cloud_loops)
    num_brep_loops = len(brep_loops)
    correspondence_matrix = np.zeros((num_stroke_cloud_loops, num_brep_loops), dtype=np.float32)

    # Step 1: Find matching between stroke_node_features and final_brep_edges
    stroke_to_brep_map = {}
    for stroke_idx, stroke in enumerate(stroke_node_features):
        stroke_points = set(map(tuple, [stroke[:3], stroke[3:6]]))
        for brep_idx, brep_edge in enumerate(final_brep_edges):
            brep_points = set(map(tuple, [brep_edge[:3], brep_edge[3:6]]))
            
            # Check if stroke is contained in brep or brep is contained in stroke
            if stroke_points.issubset(brep_points) or brep_points.issubset(stroke_points):
                if stroke_idx not in stroke_to_brep_map:
                    stroke_to_brep_map[stroke_idx] = set()
                stroke_to_brep_map[stroke_idx].add(brep_idx)  # Allow multiple matches

    # Step 2: Find the corresponding loops based on matching edges
    stroke_matched = [False] * num_stroke_cloud_loops  # Track matched stroke loops

    for i, stroke_loop in enumerate(stroke_cloud_loops):
        stroke_brep_indices = set()
        for stroke_idx in stroke_loop:
            if stroke_idx in stroke_to_brep_map:
                stroke_brep_indices.update(stroke_to_brep_map[stroke_idx])
        
        for j, brep_loop in enumerate(brep_loops):
            if stroke_brep_indices and stroke_brep_indices.issubset(set(brep_loop)):
                correspondence_matrix[i, j] = 1.0
                stroke_matched[i] = True

    return correspondence_matrix


def vis_specific_loop(loop, strokes):
    """
    Visualize specific loops and strokes.
    
    Parameters:
    loop (list of int): A list containing indices of the strokes to be highlighted.
    strokes (np.ndarray): A matrix of shape (num_strokes, 7), where the first 6 columns represent two 3D points.
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    # Initialize the 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot all strokes in blue
    for stroke in strokes:
        start, end = stroke[:3], stroke[3:6]
        ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], color='blue', alpha=0.5)

    # Plot strokes in the loop in red
    for idx in loop:
        stroke = strokes[idx]
        start, end = stroke[:3], stroke[3:6]
        ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], color='red', linewidth=2)

    # Set labels and show plot
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()


def vis_partial_graph(loops, strokes):
    """
    Visualize multiple loops and strokes in 3D space.

    Parameters:
    loops (list of lists of int): A list of loops, where each loop is a list containing indices of strokes to be highlighted.
    strokes (np.ndarray): A matrix of shape (num_strokes, 7), where the first 6 columns represent two 3D points.
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import numpy as np

    # Initialize the 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot all strokes in blue
    for stroke in strokes:
        start, end = stroke[:3], stroke[3:6]
        ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], color='blue', alpha=0.5)

    # Plot strokes in each loop with different colors
    colors = plt.cm.jet(np.linspace(0, 1, len(loops)))  # Generate a color map for the loops

    for i, loop in enumerate(loops):
        for idx in loop:
            stroke = strokes[idx]
            start, end = stroke[:3], stroke[3:6]
            ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], color=colors[i], linewidth=2)

    # Set axis labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Show plot
    plt.show()
    


def vis_brep(brep):
    """
    Visualize the brep strokes in 3D space if brep is not empty.
    
    Parameters:
    brep (np.ndarray): A matrix with shape (num_strokes, 6) representing strokes.
                       Each row contains two 3D points representing the start and end of a stroke.
                       If brep.shape[0] == 0, the function returns without plotting.
    """
    # Check if brep is empty
    if brep.shape[0] == 0:
        return

    # Initialize the 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.grid(False)

    # Initialize min and max limits
    x_min, x_max = float('inf'), float('-inf')
    y_min, y_max = float('inf'), float('-inf')
    z_min, z_max = float('inf'), float('-inf')

    # Plot all brep strokes in blue with line width 1
    for stroke in brep:
        start, end = stroke[:3], stroke[3:6]
        
        # Update the min and max limits for each axis
        x_min, x_max = min(x_min, start[0], end[0]), max(x_max, start[0], end[0])
        y_min, y_max = min(y_min, start[1], end[1]), max(y_max, start[1], end[1])
        z_min, z_max = min(z_min, start[2], end[2]), max(z_max, start[2], end[2])
        
        ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], color='blue', linewidth=1)

    # Compute the center of the shape
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    z_center = (z_min + z_max) / 2

    # Compute the maximum difference across x, y, z directions
    max_diff = max(x_max - x_min, y_max - y_min, z_max - z_min)

    # Set the same limits for x, y, and z axes centered around the computed center
    ax.set_xlim([x_center - max_diff / 2, x_center + max_diff / 2])
    ax.set_ylim([y_center - max_diff / 2, y_center + max_diff / 2])
    ax.set_zlim([z_center - max_diff / 2, z_center + max_diff / 2])

    # Set axis labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Show plot
    plt.show()
