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
import Models.sketch_model_helper


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
    
    return processed_features



#----------------------------------------------------------------------------------#


def face_to_stroke(stroke_features):
    valid_groups = face_aggregate(stroke_features)
    stroke_indices_per_face = []

    for face in valid_groups:
        face_indices = []
        for stroke in face:
            # Find the index of the stroke in stroke_features
            for i, stroke_feature in enumerate(stroke_features):
                if np.array_equal(stroke, stroke_feature):
                    face_indices.append(i)
                    break
        stroke_indices_per_face.append(face_indices)

    return stroke_indices_per_face


def face_aggregate(stroke_features):
    """
    This function permutes all the strokes and groups them into groups of 3 or 4.

    Parameters:
    stroke_matrix (numpy.ndarray): A matrix of shape (num_strokes, 6) where each row represents a stroke
                                   with start and end points in 3D space.

    Returns:
    list: A list of groups of strokes, where each group contains either 3 or 4 strokes.
    """
    
    # Ensure input is a numpy array
    stroke_matrix = np.array(stroke_features)
    
    # Reshape the stroke matrix to remove the leading dimension of 1
    stroke_matrix = stroke_matrix.reshape(-1, 6)
    
    # Get the number of strokes
    num_strokes = stroke_matrix.shape[0]
    
    # Generate all combinations of groups of 3 and 4 strokes
    groups_of_3 = list(combinations(stroke_matrix, 3))
    groups_of_4 = list(combinations(stroke_matrix, 4))
    
    # Combine the groups into a single list
    all_groups = groups_of_3 + groups_of_4
    
    def are_strokes_coplanar(group):
        for dim in range(3):  # Check each of x, y, z
            start_points = group[:, dim]
            end_points = group[:, dim + 3]
            if np.all(start_points == start_points[0]) and np.all(end_points == end_points[0]):
                return True
        return False
    
    def are_strokes_connected(group):
        points = np.concatenate([group[:, :3], group[:, 3:]], axis=0)
        unique_points, counts = np.unique(points, axis=0, return_counts=True)
        return np.all(counts == 2)

    # Filter out groups that are not coplanar
    coplanar_groups = [group for group in all_groups if are_strokes_coplanar(np.array(group))]
    valid_groups = [group for group in coplanar_groups if are_strokes_connected(np.array(group))]

    return valid_groups


def brep_to_stroke(face_feature_gnn_list, edge_features):
    stroke_features_list = [tuple(round(coord, 3) for coord in edge)  for edge in edge_features]
    face_to_stroke_indices = []

    for face in face_feature_gnn_list:
        face_indices = []
        for element in face:
            point1 = tuple(round(coord, 3) for coord in element[:3])
            point2 = tuple(round(coord, 3) for coord in element[3:])
            
            found_index = -1
            for idx, stroke in enumerate(stroke_features_list):
                stroke_point1 = stroke[:3]
                stroke_point2 = stroke[3:]
                if (point1 == stroke_point1 and point2 == stroke_point2) or (point1 == stroke_point2 and point2 == stroke_point1):
                    found_index = idx
                    break
            
            face_indices.append(found_index)
        face_to_stroke_indices.append(face_indices)
    
    return face_to_stroke_indices


def gnn_edges(brep_to_stroke):
    num_faces = len(brep_to_stroke)
    edge_matrix = np.zeros((num_faces, num_faces), dtype=np.float32)

    for i in range(num_faces):
        for j in range(num_faces):
            if i == j or any(index in brep_to_stroke[j] for index in brep_to_stroke[i]):
                edge_matrix[i, j] = 1
    
    return edge_matrix
    

def stroke_to_brep(face_to_stroke, brep_to_stroke, node_features, brep_edge_features):
    num_faces = len(face_to_stroke)
    num_breps = len(brep_to_stroke)

    result_matrix = np.zeros((num_faces, num_breps), dtype=int)
    
    node_features = np.round(node_features, 3)
    brep_edge_features = np.round(brep_edge_features, 3)
    
    for j, brep_indices in enumerate(brep_to_stroke):
        brep_lines = [brep_edge_features[idx] for idx in brep_indices]
        polygon = []

        for line in brep_lines:
            polygon.append(line)
            
        for i, face_indices in enumerate(face_to_stroke):
            face_lines = [node_features[idx] for idx in face_indices]
            points = []

            all_points_inside = True
            for line in face_lines:
                points.append(line[:3])
                points.append(line[3:])

            for point in points:
                valid_plane, plane_type, plane_value = on_same_plane(point, polygon)
                
                if valid_plane:
                    if not is_point_in_polygon(point, polygon, plane_type):
                        all_points_inside = False
                        break
                else:
                    all_points_inside = False
                    break
            
            if all_points_inside:
                result_matrix[i, j] = 1

    all_columns_connected = np.all(result_matrix.sum(axis=0) >= 1)
    return result_matrix


def coplanar_matrix(face_to_stroke, node_features):
    num_faces = len(face_to_stroke)
    coplanar_matrix = np.zeros((num_faces, num_faces), dtype=int)

    face_planes = []

    for face_indices in face_to_stroke:
        points = []
        for idx in face_indices:
            stroke = node_features[idx]
            points.append(stroke[:3])
            points.append(stroke[3:])

        points = np.array(points)
        unique_x = np.unique(points[:, 0])
        unique_y = np.unique(points[:, 1])
        unique_z = np.unique(points[:, 2])

        if len(unique_x) == 1:
            face_planes.append(('x', unique_x[0]))
        elif len(unique_y) == 1:
            face_planes.append(('y', unique_y[0]))
        elif len(unique_z) == 1:
            face_planes.append(('z', unique_z[0]))
        else:
            face_planes.append(('none', unique_x[0]))

    for i in range(num_faces):
        for j in range(i, num_faces):
            if face_planes[i] == face_planes[j]:
                coplanar_matrix[i, j] = 1
                coplanar_matrix[j, i] = 1

    return coplanar_matrix
#----------------------------------------------------------------------------------#

def get_plane(polygon):
    # Extract the unique x, y, and z values from the polygon
    x_values = set()
    y_values = set()
    z_values = set()
    
    for line in polygon:
        x_values.add(line[0])
        y_values.add(line[1])
        z_values.add(line[2])
        x_values.add(line[3])
        y_values.add(line[4])
        z_values.add(line[5])
    
    # Determine the plane type and the value
    if len(x_values) == 1:
        return 'x', next(iter(x_values))
    elif len(y_values) == 1:
        return 'y', next(iter(y_values))
    elif len(z_values) == 1:
        return 'z', next(iter(z_values))
    else:
        return None, None


def on_same_plane(point, polygon):
    # Extract the unique x, y, and z values from the polygon
    x_values = set()
    y_values = set()
    z_values = set()
    
    for line in polygon:
        x_values.add(line[0])
        y_values.add(line[1])
        z_values.add(line[2])
        x_values.add(line[3])
        y_values.add(line[4])
        z_values.add(line[5])
    
    # Check if the polygon is on a specific plane and if the point is on that plane
    if len(x_values) == 1 and point[0] in x_values:
        return True, 'x', point[0]
    elif len(y_values) == 1 and point[1] in y_values:
        return True, 'y', point[1]
    elif len(z_values) == 1 and point[2] in z_values:
        return True, 'z', point[2]
    else:
        return False, None, None





def is_point_in_polygon(point, polygon, plane_type):
    # Extract 2D polygon based on the plane_type
    if plane_type == 'x':
        polygon_2d = [(line[1], line[2], line[4], line[5]) for line in polygon]
        point_2d = (point[1], point[2])
    elif plane_type == 'y':
        polygon_2d = [(line[0], line[2], line[3], line[5]) for line in polygon]
        point_2d = (point[0], point[2])
    elif plane_type == 'z':
        polygon_2d = [(line[0], line[1], line[3], line[4]) for line in polygon]
        point_2d = (point[0], point[1])
    else:
        raise ValueError("Invalid plane_type. Must be 'x', 'y', or 'z'.")

    # Helper function to check if point is on a line segment
    def is_point_on_line(px, py, x0, y0, x1, y1):
        if min(x0, x1) <= px <= max(x0, x1) and min(y0, y1) <= py <= max(y0, y1):
            if (x1 - x0) * (py - y0) == (y1 - y0) * (px - x0):
                return True
        return False

    # Ray casting algorithm to check if point is in polygon
    def is_point_in_2d_polygon(point, polygon_2d):
        x, y = point
        n = len(polygon_2d)
        inside = False

        for i in range(n):
            x0, y0, x1, y1 = polygon_2d[i]
            
            # Check if the point is on a vertex
            if (x, y) == (x0, y0) or (x, y) == (x1, y1):
                return True

            # Check if the point is on an edge
            if is_point_on_line(x, y, x0, y0, x1, y1):
                return True

            # Ray casting algorithm
            if ((y0 > y) != (y1 > y)) and (x < (x1 - x0) * (y - y0) / (y1 - y0) + x0):
                inside = not inside
        
        return inside

    return is_point_in_2d_polygon(point_2d, polygon_2d)
