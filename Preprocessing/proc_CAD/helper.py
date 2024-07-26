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

    