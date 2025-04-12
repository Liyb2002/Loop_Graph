import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
from scipy.optimize import least_squares
from scipy.interpolate import splprep, splev, CubicSpline

from itertools import combinations

from itertools import product


import json
import os




# ------------------------------------------------------------------------------------# 




def build_final_edges_json(final_edges_json):
    node_features_list = []

    num_edges = len(final_edges_json)
    is_feature_line_matrix = np.zeros((num_edges, 1))

    for i, key in enumerate(final_edges_json.keys()):
        stroke = final_edges_json[key]

        geometry = stroke["geometry"]

        node_feature = build_node_features(geometry)


        node_features_list.append(node_feature)

        stroke_type = stroke['type']
        if stroke_type in ['feature_line', 'extrude_line', 'fillet_line']:
            is_feature_line_matrix[i] = 1
        else:
            is_feature_line_matrix[i] = 0


    node_features_matrix = np.array(node_features_list)

    return node_features_matrix, is_feature_line_matrix



def build_all_edges_json(all_edges_json):
    node_features_list = []

    for stroke in all_edges_json:
        geometry = stroke["geometry"]

        node_feature = build_node_features(geometry)
        
        node_features_list.append(node_feature)

    node_features_matrix = np.array(node_features_list)

    return node_features_matrix


# ------------------------------------------------------------------------------------# 


# Straight Line: 10 values + type 1
# 0-2: point1, 3-5:point2, 6:alpha_value, 7-9: 0

# Circle Feature: 10 values + type 2
# 0-2: center, 3-5:normal, 6:alpha_value, 7:radius, 8-9: 0

# Arc Feature: 10 values + type 3
# 0-2: point1, 3-5:point2, 6:alpha_value, 7-9:center

# Ellipse Feature: 10 values + type 4
# 0-2: center, 3-5:normal, 6:alpha_value, 7: major axis, 8: minor axis, 9: orientation

# Closed Line: 10 values + type 5
# 0-2: point1, 3-5: point2, 6:alpha_value, 7-9: random point in the line

# Curved Line: 10 values + type 6
# 0-2: point1, 3-5: point2, 6:alpha_value, 7-9: random point in the line


def build_node_features(geometry):
    num_points = len(geometry)
    alpha_value = 0

    # Case 1: Check if the geometry has low residual fitting straight line  -> (Straight Line)
    residual = fit_straight_line(geometry)
    threshold = dist(geometry) * 2

    if residual < threshold:
        point1 = geometry[0]
        point2 = geometry[-1]

        return point1 + point2 + [alpha_value] + [0, 0, 0, 1]

    # Check if geometry is closed
    distance, closed = is_closed_shape(geometry)

    if not closed or len(geometry) < 5:
        center_circle, radius_circle, normal_circle, circle_residual = fit_circle_3d(geometry)

        if circle_residual < threshold:
            # Case 3: Arc
            point1 = geometry[0]
            point2 = geometry[-1]
            return point1 + point2 + [alpha_value] + center_circle + [3]

        # Case 6: Curved Line
        point1 = geometry[0]
        point2 = geometry[-1]
        random_point = geometry[len(geometry) // 2]
        return point1 + point2 + [alpha_value] + random_point + [6]

    # Try fitting a circle
    center_circle, radius_circle, normal_circle, circle_residual = fit_circle_3d(geometry)

    if circle_residual > threshold:
        # Case 5: Closed Shape
        point1 = geometry[0]
        point2 = geometry[-1]
        random_point = geometry[len(geometry) // 2]

        return point1 + point2 + [alpha_value] + random_point + [5]



    # Try fitting an ellipse
    center_ellipse, normal_ellipse, axes_lengths, theta, ellipse_residual = fit_ellipse_3d(geometry)
    major_axis, minor_axis = axes_lengths

    if abs(major_axis - minor_axis) < threshold:
        # Case 4: Ellipse
        return center_ellipse + normal_ellipse+ [alpha_value] + [major_axis, minor_axis, 0, 4]
    
    # Case 2: Circle

    # print("geometry", geometry)
    # print("center_circle", center_circle)
    # print("---------------")
    return center_circle + normal_circle +  [alpha_value] + [radius_circle, 0, 0, 2]



# Brep edge format:
# 1)Straight Line: Point_1 (3 value), Point_2 (3 value), 0, 0, 0, 1
# 2)Cicles: Center (3 value), normal (3 value), 0, radius, 0, 2
# 3)Cylinder face: Center (3 value), normal (3 value), height, radius, 0, 3
# 4)Arc: Point_1 (3 value), Point_2 (3 value), Center (3 value), 4



def rotate_matrix(edge_features_list, cylinder_features, rotation_matrix):
    rotated_edges = []
    rotated_cylinder_edges = []

    rotation_matrix = np.array(rotation_matrix)


    scale_x = np.linalg.norm(rotation_matrix[:3, 0])
    scale_y = np.linalg.norm(rotation_matrix[:3, 1])
    scale_z = np.linalg.norm(rotation_matrix[:3, 2])
    average_scale = (scale_x + scale_y + scale_z) / 3


    # Rotate edge endpoints
    for edge in edge_features_list:
        point_1 = np.array(edge[0:3] + [1.0])
        point_2 = np.array(edge[3:6] + [1.0])

        rotated_point_1 = rotation_matrix @ point_1
        rotated_point_2 = rotation_matrix @ point_2

        rotated_edge = rotated_point_1[:3].tolist() + rotated_point_2[:3].tolist() + edge[6:]
        rotated_edges.append(rotated_edge)

    # Rotate cylinder centers
    for cylinder_edge in cylinder_features:
        center = np.array(cylinder_edge[0:3] + [1.0])  # Assuming center is at [0:3]
        height = cylinder_edge[6]
        radius = cylinder_edge[7]
        scaled_height = height * average_scale
        scaled_radius = radius * average_scale

        rotated_center = rotation_matrix @ center
        rotated_cylinder = rotated_center[:3].tolist() + cylinder_edge[3:6] + [scaled_height, scaled_radius] + cylinder_edge[8:]
        rotated_cylinder_edges.append(rotated_cylinder)

    return rotated_edges, rotated_cylinder_edges

# ------------------------------------------------------------------------------------# 



def build_stroke_operation_matrix(final_edges_data):
    # Step 1: Get the number of strokes and unique feature_ids
    num_strokes = len(final_edges_data)
    feature_ids = set()  # to store all unique feature_ids

    # Step 2: Collect all unique feature_ids from final_edges_data
    for stroke in final_edges_data.values():
        feature_ids.add(stroke['feature_id'])

    # Step 3: Convert the set of feature_ids to a sorted list
    feature_ids = sorted(feature_ids)

    # Step 4: Create the matrix with shape (num_strokes, num_feature_ids)
    matrix = np.zeros((num_strokes, len(feature_ids)), dtype=int)

    # Step 5: Fill the matrix
    for i, (key, stroke) in enumerate(final_edges_data.items()):
        feature_id = stroke['feature_id']
        feature_id_index = feature_ids.index(feature_id)  # find the index of the feature_id
        matrix[i, feature_id_index] = 1  # Mark this position as 1 for the correct feature_id

    return matrix


# ------------------------------------------------------------------------------------# 


def find_new_features(prev_brep_edges, new_edge_features):
    prev_brep_edges = [[round(coord, 4) for coord in line] for line in prev_brep_edges]
    new_edge_features = [[round(coord, 4) for coord in line] for line in new_edge_features]

    def is_same_direction(line1, line2):
        """Check if two lines have the same direction."""
        vector1 = np.array(line1[3:6]) - np.array(line1[:3])
        vector2 = np.array(line2[3:6]) - np.array(line2[:3])
        return np.allclose(vector1 / np.linalg.norm(vector1), vector2 / np.linalg.norm(vector2))

    def is_point_on_line(point, line):
        """Check if a point lies on a given line segment."""
        start, end = np.array(line[:3]), np.array(line[3:6])

        # Check if the point is collinear (still important to check)
        if not np.allclose(np.cross(end - start, point - start), 0):
            return False
        
        # Check if point lies within the bounds of the line segment
        min_x, max_x = min(start[0], end[0]), max(start[0], end[0])
        min_y, max_y = min(start[1], end[1]), max(start[1], end[1])
        min_z, max_z = min(start[2], end[2]), max(start[2], end[2])
        
        return (min_x <= point[0] <= max_x) and (min_y <= point[1] <= max_y) and (min_z <= point[2] <= max_z)

    def is_line_contained(line1, line2):
        """Check if line1 is contained within line2."""
        return is_point_on_line(np.array(line1[:3]), line2) and is_point_on_line(np.array(line1[3:6]), line2)



    def find_unique_points(new_edge_line, prev_brep_line):
        """Find the two unique points between new_edge_line and prev_brep_line."""
        points = [
            tuple(new_edge_line[:3]),   # new_edge_line start
            tuple(new_edge_line[3:6]),   # new_edge_line end
            tuple(prev_brep_line[:3]),  # prev_brep_line start
            tuple(prev_brep_line[3:6]),  # prev_brep_line end
        ]

        # Find unique points
        unique_points = [point for point in points if points.count(point) == 1]

        # Ensure there are exactly two unique points
        if len(unique_points) == 2:
            return unique_points
        return None

    new_features = []

    for new_edge_line in new_edge_features:
        if new_edge_line[-1] != 0:
            new_features.append(new_edge_line)
            continue

        relation_found = False

        edge_start, edge_end = np.array(new_edge_line[:3]), np.array(new_edge_line[3:6])
        
        for prev_brep_line in prev_brep_edges:
            if prev_brep_line[-1] != 0:
                continue

            brep_start, brep_end = np.array(prev_brep_line[:3]), np.array(prev_brep_line[3:6])

            # This is a circle edge
            if (edge_start == edge_end).all():
                break

            # Check if the lines are the same, either directly or in reverse order
            if (np.allclose(edge_start, brep_start) and np.allclose(edge_end, brep_end)) or \
            (np.allclose(edge_start, brep_end) and np.allclose(edge_end, brep_start)):
                # Relation 1: The two lines are exactly the same
                relation_found = True
                break
            
            elif is_same_direction(new_edge_line, prev_brep_line) and is_line_contained(new_edge_line, prev_brep_line):
                # new feature is in prev brep
                relation_found = True
                
                unique_points = find_unique_points(new_edge_line, prev_brep_line)
                if unique_points:
                    # Create a new line using the unique points
                    new_line = list(unique_points[0]) + list(unique_points[1])
                    new_features.append(new_line)
                    relation_found = True
                    break
                break
            
            elif is_same_direction(new_edge_line, prev_brep_line) and is_line_contained(prev_brep_line, new_edge_line):
                # prev brep is in new feature
                relation_found = True
                
                unique_points = find_unique_points(new_edge_line, prev_brep_line)
                if unique_points:
                    # Create a new line using the unique points
                    new_line = list(unique_points[0]) + list(unique_points[1])
                    new_features.append(new_line)
                    relation_found = True
                    break

                break
        
        if not relation_found:
            # Relation 4: None of the relations apply
            new_features.append(new_edge_line)

    return new_features



import numpy as np

def is_line_contained(container_line, test_line, tol=1e-7):
    """
    Returns True if the bounding box of test_line is fully contained
    within the bounding box of container_line.
    
    container_line and test_line are each [x1, y1, z1, x2, y2, z2].
    tol is used to allow for small floating-point variations.
    
    NOTE: This only checks bounding-box containment, not strict collinearity.
    In other words, as long as test_line’s endpoints fall within container_line’s
    min/max x, y, z ranges (within the tolerance), we consider it "contained."
    """
    # Convert lines to arrays
    c1 = np.array(container_line[:3])
    c2 = np.array(container_line[3:6])
    t1 = np.array(test_line[:3])
    t2 = np.array(test_line[3:6])
    
    # Compute min/max for container_line
    c_min = np.minimum(c1, c2)
    c_max = np.maximum(c1, c2)
    
    # Compute min/max for test_line
    t_min = np.minimum(t1, t2)
    t_max = np.maximum(t1, t2)
    
    # Check all coordinates are inside container_line’s bounding box (with tolerance)
    if np.all(t_min >= c_min - tol) and np.all(t_max <= c_max + tol):
        return True
    else:
        return False


def find_new_features_simple(prev_brep_edges, new_edge_features):
    """
    - Returns a list of unique edges that do NOT exactly match any edge in prev_brep_edges.
    - Also returns a set of indices for any prev_brep_edges that are used, i.e.:
      either matched exactly or found to be contained in a new edge.
    """
    unique_edges = []
    used_prev_edges = set()  # store indices of prev_brep_edges considered used
    
    for new_edge_line in new_edge_features:
        edge_start = np.array(new_edge_line[:3])
        edge_end   = np.array(new_edge_line[3:6])
        
        is_unique = True  # assume new_edge_line is unique unless we find an exact match
        
        for i, prev_brep_line in enumerate(prev_brep_edges):
            brep_start = np.array(prev_brep_line[:3])
            brep_end   = np.array(prev_brep_line[3:6])
            
            # 1) Check if new edge exactly matches an existing one (forward or reversed).
            if (np.allclose(edge_start, brep_start) and np.allclose(edge_end, brep_end)) or \
               (np.allclose(edge_start, brep_end)   and np.allclose(edge_end, brep_start)):
                # The new edge is not unique; also consider the old edge "used"
                is_unique = False
                break
            
            # 2) Check if the new edge *contains* this previous edge
            # if is_line_contained(prev_brep_line, new_edge_line):
            #     is_unique = False
            #     break
        
        # After comparing with all prev edges:
        if is_unique:
            unique_edges.append(new_edge_line)
    
    return unique_edges



# ------------------------------------------------------------------------------------# 
def fit_straight_line(points):
    """
    Fits a straight line to a set of 3D points by using the first and last points as endpoints.

    Parameters:
        points: numpy array of shape (N, 3), where N is the number of points.

    Returns:
        start_point: The first point (start of the line).
        end_point: The last point (end of the line).
        avg_residual: The average distance of all points to the line.
    """
    points = np.array(points)

    start_point = points[0]
    end_point = points[-1]

    direction = end_point - start_point
    direction /= np.linalg.norm(direction) 

    residuals = []
    for point in points:
        vector_to_point = point - start_point
        projection_length = np.dot(vector_to_point, direction)
        projected_point = start_point + projection_length * direction
        residual = np.linalg.norm(point - projected_point)
        residuals.append(residual)

    avg_residual = np.mean(residuals)

    return avg_residual


def fit_circle_3d(points):
    """
    Fit a circle directly in 3D space using non-linear least squares optimization.
    The normal vector is pre-computed and used to simplify the fitting process.

    Parameters:
        points (np.ndarray): An (N, 3) array of 3D points.

    Returns:
        center (np.ndarray): The center of the fitted circle.
        radius (float): The radius of the fitted circle.
        normal (np.ndarray): The normal vector of the fitted circle.
        mean_residual (float): The mean residual of the fit.
    """
    
    # Pre-compute the normal using the given points
    normal = compute_normal(points)

    
    def residuals(params, points, normal):
        """
        Compute residuals (distances) from the points to the circle defined by params.
        
        Parameters:
            params: [x_c, y_c, z_c, radius]
                - (x_c, y_c, z_c): Center of the circle
                - radius: Radius of the circle
            points: The input 3D points.
            normal: The normal vector of the plane.
        Returns:
            Residuals as distances from the points to the circle.
        """
        center = params[:3]
        radius = params[3]
        
        # Normalize the normal vector to ensure it has unit length
        normal = normal / np.linalg.norm(normal)
        
        # Calculate vector from center to each point
        vecs = points - center
        
        # Check if any input is invalid
        if not np.isfinite(vecs).all() or not np.isfinite(normal).all():
            return np.full(len(points), 1e10)  # Return a large value if inputs are invalid

        # Project the vectors onto the plane defined by the normal
        dot_products = np.dot(vecs, normal)
        vecs_proj = vecs - np.outer(dot_products, normal)
        
        # Calculate distances from the projected points to the circle's radius
        distances = np.linalg.norm(vecs_proj, axis=1) - radius
        
        # Replace any NaNs or infinite values with a large number
        distances = np.nan_to_num(distances, nan=1e10, posinf=1e10, neginf=-1e10)

        return distances

    # Step 1: Estimate initial parameters
    center_init = np.mean(points, axis=0)
    radius_init = np.mean(np.linalg.norm(points - center_init, axis=1))
    params_init = np.hstack([center_init, radius_init])

    # Step 2: Use least squares optimization to fit the circle
    result = least_squares(residuals, params_init, args=(points, normal))
    
    # Extract optimized parameters
    center_opt = result.x[:3]
    radius_opt = result.x[3]
    final_residuals = residuals(result.x, points, normal)

    # print("points", points)
    # print("center_init", center_init)
    # print("-------")
    # print("center_opt:", center_opt)
    # print("radius_opt", radius_opt)
    # print("normal:", normal)
    # print("np.mean(np.abs(final_residuals))", np.mean(np.abs(final_residuals)))
    # print("-------")

    return list(center_init), radius_opt, list(normal), np.mean(np.abs(final_residuals))



def check_if_arc(points, center, radius, normal):
    # Step 1: Calculate vectors from the center to each point
    vecs = points - center
    
    # Step 2: Project the vectors onto the plane defined by the normal vector
    vecs_proj = vecs - np.outer(np.dot(vecs, normal), normal)
    
    # Step 3: Calculate angles of the projected points relative to a reference vector
    ref_vector = vecs_proj[0] / np.linalg.norm(vecs_proj[0])
    angles = np.arctan2(
        np.dot(vecs_proj, np.cross(normal, ref_vector)),
        np.dot(vecs_proj, ref_vector)
    )
    
    # Normalize angles to [0, 2*pi]
    angles = (angles + 2 * np.pi) % (2 * np.pi)
    
    # Step 4: Calculate the angular range covered by the points
    min_angle = np.min(angles)
    max_angle = np.max(angles)
    raw_angle = max_angle - min_angle
    angle_range = min(raw_angle, (6.28-raw_angle))
    
    # Step 5: Determine if the points form an arc or a full circle
    is_arc = angle_range < 2 * np.pi - 0.01  # Allow a small tolerance for numerical errors
    return angle_range, is_arc



def fit_ellipse_3d(points):
    
    def residuals(params, points):
        center = params[:3]
        normal = params[3:6]
        a = params[6]
        b = params[7]
        theta = params[8]
        
        # Normalize the normal vector
        normal = normal / np.linalg.norm(normal)
        
        # Calculate vectors from center to each point
        vecs = points - center
        
        # Project vectors onto the plane defined by the normal vector
        vecs_proj = vecs - np.outer(np.dot(vecs, normal), normal)
        
        # Define the major and minor axis direction vectors in the plane
        major_axis_dir = np.array([np.cos(theta), np.sin(theta), 0])
        minor_axis_dir = np.array([-np.sin(theta), np.cos(theta), 0])
        
        # Project onto the ellipse axes
        x_proj = np.dot(vecs_proj, major_axis_dir)
        y_proj = np.dot(vecs_proj, minor_axis_dir)
        
        # Compute the residuals using the ellipse equation
        residuals = (x_proj / a)**2 + (y_proj / b)**2 - 1
        return residuals

    # Step 1: Use fit_circle_3d to find an initial estimate for the plane
    center_init, _, normal_init, _ = fit_circle_3d(points)
    center_init = np.array(center_init)
    normal_init = np.array(normal_init)

    
    # Step 2: Estimate initial parameters for the ellipse
    a_init = np.max(np.linalg.norm(points - center_init, axis=1))
    b_init = a_init * 0.5  # Initial guess for minor axis
    theta_init = 0.0  # Initial guess for the orientation
    params_init = np.hstack([center_init, normal_init, a_init, b_init, theta_init])

    # Step 3: Use least squares optimization to fit the ellipse in 3D
    result = least_squares(residuals, params_init, args=(points,))
    
    # Extract optimized parameters
    center_opt = result.x[:3]
    normal_opt = result.x[3:6]
    a_opt = result.x[6]
    b_opt = result.x[7]
    theta_opt = result.x[8]
    
    # Normalize the normal vector
    normal_opt = normal_opt / np.linalg.norm(normal_opt)
    
    # Calculate major and minor axis directions
    major_axis_dir = np.array([np.cos(theta_opt), np.sin(theta_opt), 0])
    minor_axis_dir = np.array([-np.sin(theta_opt), np.cos(theta_opt), 0])
    
    # Calculate the mean residual
    final_residuals = residuals(result.x, points)
    mean_residual = np.mean(np.abs(final_residuals))
    
    return list(center_opt), list(normal_opt), (a_opt, b_opt), theta_opt, mean_residual



def is_closed_shape(points):
    points = np.array(points)
    distance = np.linalg.norm(points[0] - points[-1])
    tolerance = dist(points) * 2
    

    return distance, distance < tolerance


def dist(points):
    points = np.array(points)
    distances = np.linalg.norm(points[1:] - points[:-1], axis=1)
    average_distance = np.mean(distances)
    
    return average_distance


def compute_normal(points):
    points = np.array(points)

    A = points[0]
    B = points[len(points) // 2]
    C = points[len(points) // 4]
    
    AB = B - A
    AC = C - A
    
    normal = np.cross(AB, AC)
    
    normal /= np.linalg.norm(normal)
    
    return normal

# ------------------------------------------------------------------------------------# 


import numpy as np

def is_colinear(p1, p2, p3, tol=1e-6):
    """Check if 3 points are colinear."""
    v1 = p2 - p1
    v2 = p3 - p1
    cross_prod = np.cross(v1, v2)
    return np.linalg.norm(cross_prod) < tol


# ------------------------------------------------------------------------------------# 
def from_stroke_to_edge(stroke_to_edge, stroke_cloud_loops):
    loop_choices = []

    for loop in stroke_cloud_loops:
        chosen_count = sum(stroke_to_edge[idx, 0] == 1 for idx in loop)
        if chosen_count >= len(loop):
            loop_choices.append([1])
        else:
            loop_choices.append([0])

    return np.array(loop_choices, dtype=int)



# ------------------------------------------------------------------------------------# 


def build_intersection_matrix(strokes_dict_data):
    """
    Builds an intersection matrix indicating which strokes intersect with others.
    
    Parameters:
    - strokes_dict_data (list): A list of dictionaries where each dictionary represents a stroke 
      and contains an 'intersections' key, which is a list of sublists with intersecting stroke indices.

    Returns:
    - numpy.ndarray: A matrix of shape (num_strokes_dict_data, num_strokes_dict_data),
      where a value of 1 indicates that a stroke intersects another stroke in a one-way manner.
    """
    num_strokes = len(strokes_dict_data)
    intersection_matrix = np.zeros((num_strokes, num_strokes), dtype=np.int32)  # Initialize with 0s

    for idx, stroke_dict in enumerate(strokes_dict_data):
        intersect_strokes = stroke_dict.get("intersections", [])  # Get intersection lists
        
        # Unfold the sublists to get all intersecting stroke indices
        intersecting_indices = {stroke_idx for sublist in intersect_strokes for stroke_idx in sublist}

        # Mark intersections in the matrix (acyclic, so only row updates)
        for intersecting_idx in intersecting_indices:
            if 0 <= intersecting_idx < num_strokes:  # Ensure index is valid
                intersection_matrix[idx, intersecting_idx] = 1  # One-way intersection

    return intersection_matrix


# ------------------------------------------------------------------------------------# 



def vis_feature_lines(feature_lines):
    """
    Visualize only the feature_line strokes in 3D space without axes, background, or automatic zooming.

    Parameters:
    - feature_lines (list): List of stroke dictionaries containing geometry (list of 3D points).
    """
    # Initialize the 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Remove axis labels, ticks, and background
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_frame_on(False)
    ax.grid(False)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.set_axis_off()  # Hides the axis completely

    # Initialize bounding box variables
    x_min, x_max = float('inf'), float('-inf')
    y_min, y_max = float('inf'), float('-inf')
    z_min, z_max = float('inf'), float('-inf')

    # Loop through all feature_line strokes
    for stroke in feature_lines:
        geometry = stroke["geometry"]

        if len(geometry) < 2:
            continue  # Ensure there are enough points to plot
            
        if len(geometry) > 2:
            for j in range(0, len(geometry)):
                pt = geometry[j]


        # Plot each segment of the stroke
        for j in range(1, len(geometry)):
            start = geometry[j - 1]
            end = geometry[j]

            # Update bounding box
            x_min, x_max = min(x_min, start[0], end[0]), max(x_max, start[0], end[0])
            y_min, y_max = min(y_min, start[1], end[1]), max(y_max, start[1], end[1])
            z_min, z_max = min(z_min, start[2], end[2]), max(z_max, start[2], end[2])

            # Extract coordinates for plotting
            x_values = [start[0], end[0]]
            y_values = [start[1], end[1]]
            z_values = [start[2], end[2]]

            # Plot the stroke as a black line
            ax.plot(x_values, y_values, z_values, color='black', linewidth=0.5)

    # Compute the center and rescale
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    z_center = (z_min + z_max) / 2
    max_diff = max(x_max - x_min, y_max - y_min, z_max - z_min)

    ax.set_xlim([x_center - max_diff / 2, x_center + max_diff / 2])
    ax.set_ylim([y_center - max_diff / 2, y_center + max_diff / 2])
    ax.set_zlim([z_center - max_diff / 2, z_center + max_diff / 2])

    # Show the plot
    plt.show()




def vis_feature_lines_selected(feature_lines, stroke_node_features, new_stroke_to_edge_matrix):
    """
    Visualize 3D strokes, with a subset of chosen strokes highlighted in red.

    Parameters:
    - feature_lines (list): List of stroke dictionaries containing geometry (list of 3D points).
    - stroke_node_features: (unused in this function, can be removed if not needed).
    - new_stroke_to_edge_matrix (np.ndarray or list): An array of shape (num_lines,)
      with value == 1 if a stroke is chosen and should be highlighted.
    """
    # Initialize the 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Remove axis labels, ticks, and background
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_frame_on(False)
    ax.grid(False)
    ax.set_axis_off()

    # Initialize bounding box variables
    x_min, x_max = float('inf'), float('-inf')
    y_min, y_max = float('inf'), float('-inf')
    z_min, z_max = float('inf'), float('-inf')

    # First pass: plot all strokes in black and compute bounding box
    for stroke in feature_lines:
        geometry = stroke["geometry"]

        if len(geometry) < 2:
            continue

        for j in range(1, len(geometry)):
            start = geometry[j - 1]
            end = geometry[j]

            # Update bounding box
            x_min, x_max = min(x_min, start[0], end[0]), max(x_max, start[0], end[0])
            y_min, y_max = min(y_min, start[1], end[1]), max(y_max, start[1], end[1])
            z_min, z_max = min(z_min, start[2], end[2]), max(z_max, start[2], end[2])

            # Plot the stroke in black
            ax.plot([start[0], end[0]], 
                    [start[1], end[1]], 
                    [start[2], end[2]], 
                    color='black', 
                    linewidth=0.5)

    # Compute the center and rescale
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    z_center = (z_min + z_max) / 2
    max_diff = max(x_max - x_min, y_max - y_min, z_max - z_min)

    ax.set_xlim([x_center - max_diff / 2, x_center + max_diff / 2])
    ax.set_ylim([y_center - max_diff / 2, y_center + max_diff / 2])
    ax.set_zlim([z_center - max_diff / 2, z_center + max_diff / 2])

    # Second pass: highlight chosen strokes in red
    for idx, selected in enumerate(new_stroke_to_edge_matrix):
        if selected == 1:
            if idx < len(feature_lines):
                geometry = feature_lines[idx]["geometry"]
                if len(geometry) < 2:
                    continue
                for j in range(1, len(geometry)):
                    start = geometry[j - 1]
                    end = geometry[j]
                    ax.plot([start[0], end[0]],
                            [start[1], end[1]],
                            [start[2], end[2]],
                            color='red',
                            linewidth=1.0)
            else:
                stroke = stroke_node_features[idx]
                start = stroke[0:3]
                end = stroke[3:6]
                ax.plot([start[0], end[0]],
                        [start[1], end[1]],
                        [start[2], end[2]],
                        color='red',
                        linewidth=1.0)

    plt.show()



def vis_feature_lines_loop_ver(feature_lines, stroke_to_loop, stroke_cloud_loops):
    """
    Visualize 3D strokes, with a subset of chosen loops' strokes highlighted in red.

    Parameters:
    - feature_lines (list): List of stroke dictionaries containing geometry (list of 3D points).
    - stroke_to_loop (np.ndarray): Array of shape (num_loops, 1), 1 indicates loop is selected.
    - stroke_cloud_loops (list of lists): Each sublist contains stroke indices forming a loop.
    """
    # Initialize the 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Remove axis labels, ticks, and background
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_frame_on(False)
    ax.grid(False)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.set_axis_off()

    # Initialize bounding box variables
    x_min, x_max = float('inf'), float('-inf')
    y_min, y_max = float('inf'), float('-inf')
    z_min, z_max = float('inf'), float('-inf')

    # First pass: draw all strokes in black
    for stroke in feature_lines:
        geometry = stroke["geometry"]
        if len(geometry) < 2:
            continue
        for j in range(1, len(geometry)):
            start, end = geometry[j - 1], geometry[j]

            # Update bounding box
            x_min, x_max = min(x_min, start[0], end[0]), max(x_max, start[0], end[0])
            y_min, y_max = min(y_min, start[1], end[1]), max(y_max, start[1], end[1])
            z_min, z_max = min(z_min, start[2], end[2]), max(z_max, start[2], end[2])

            # Plot in black
            ax.plot([start[0], end[0]],
                    [start[1], end[1]],
                    [start[2], end[2]],
                    color='black',
                    linewidth=0.5)

    # Second pass: highlight strokes from selected loops in red
    num_loops = len(stroke_cloud_loops)
    for loop_idx in range(num_loops):
        if stroke_to_loop[loop_idx] == 1:
            stroke_indices = stroke_cloud_loops[loop_idx]
            for stroke_idx in stroke_indices:
                if stroke_idx >= len(feature_lines):
                    continue
                geometry = feature_lines[stroke_idx]["geometry"]
                if len(geometry) < 2:
                    continue
                for j in range(1, len(geometry)):
                    start, end = geometry[j - 1], geometry[j]
                    ax.plot([start[0], end[0]],
                            [start[1], end[1]],
                            [start[2], end[2]],
                            color='red',
                            linewidth=1.0)

    # Normalize view
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    z_center = (z_min + z_max) / 2
    max_diff = max(x_max - x_min, y_max - y_min, z_max - z_min)

    ax.set_xlim([x_center - max_diff / 2, x_center + max_diff / 2])
    ax.set_ylim([y_center - max_diff / 2, y_center + max_diff / 2])
    ax.set_zlim([z_center - max_diff / 2, z_center + max_diff / 2])

    # Show the plot
    plt.show()




def vis_feature_lines_loop_all(feature_lines, stroke_node_features, stroke_cloud_loops):
    """
    Visualize 3D strokes, highlighting one loop at a time in red.
    A separate plot is generated for each loop.

    Parameters:
    - feature_lines (list): List of stroke dictionaries containing geometry (list of 3D points).
    - stroke_cloud_loops (list of lists): Each sublist contains stroke indices forming a loop.
    """
    import matplotlib.pyplot as plt

    # Precompute bounding box across all strokes
    x_min, x_max = float('inf'), float('-inf')
    y_min, y_max = float('inf'), float('-inf')
    z_min, z_max = float('inf'), float('-inf')

    for stroke in feature_lines:
        geometry = stroke["geometry"]
        if len(geometry) < 2:
            continue
        for j in range(1, len(geometry)):
            start, end = geometry[j - 1], geometry[j]
            x_min, x_max = min(x_min, start[0], end[0]), max(x_max, start[0], end[0])
            y_min, y_max = min(y_min, start[1], end[1]), max(y_max, start[1], end[1])
            z_min, z_max = min(z_min, start[2], end[2]), max(z_max, start[2], end[2])

    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    z_center = (z_min + z_max) / 2
    max_diff = max(x_max - x_min, y_max - y_min, z_max - z_min)

    # Loop over each loop and create a new plot
    for loop_idx, stroke_indices in enumerate(stroke_cloud_loops):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Remove axis labels, ticks, and background
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.set_frame_on(False)
        ax.grid(False)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        ax.set_axis_off()

        # Draw all strokes in black
        for stroke in feature_lines:
            geometry = stroke["geometry"]
            if len(geometry) < 2:
                continue
            for j in range(1, len(geometry)):
                start, end = geometry[j - 1], geometry[j]
                ax.plot([start[0], end[0]],
                        [start[1], end[1]],
                        [start[2], end[2]],
                        color='black',
                        linewidth=0.5)

        # Highlight current loop in red
        for stroke_idx in stroke_indices:
            if stroke_idx >= len(feature_lines):
                # Fallback to stroke_node_features
                stroke = stroke_node_features[stroke_idx]
                point_1 = stroke[:3]
                point_2 = stroke[3:6]
                ax.plot([point_1[0], point_2[0]],
                        [point_1[1], point_2[1]],
                        [point_1[2], point_2[2]],
                        color='red',
                        linewidth=1.0)
                continue

            geometry = feature_lines[stroke_idx]["geometry"]
            if len(geometry) < 2:
                continue
            for j in range(1, len(geometry)):
                start, end = geometry[j - 1], geometry[j]
                ax.plot([start[0], end[0]],
                        [start[1], end[1]],
                        [start[2], end[2]],
                        color='red',
                        linewidth=1.0)

        # Set limits
        ax.set_xlim([x_center - max_diff / 2, x_center + max_diff / 2])
        ax.set_ylim([y_center - max_diff / 2, y_center + max_diff / 2])
        ax.set_zlim([z_center - max_diff / 2, z_center + max_diff / 2])

        # Optional: title
        ax.set_title(f"Loop {loop_idx}", pad=20)

        # Show current plot
        plt.show()




def vis_stroke_node_features(stroke_node_features):
    # Initialize the 3D plot

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.grid(False)
    ax.axis('off')  # Turn off axis background and borders

    # Initialize min and max limits
    x_min, x_max = float('inf'), float('-inf')
    y_min, y_max = float('inf'), float('-inf')
    z_min, z_max = float('inf'), float('-inf')

    perturb_factor = 0.000002  # Adjusted perturbation factor for hand-drawn effect

    # Plot all strokes in blue with perturbations
    for idx, stroke in enumerate(stroke_node_features):
        start, end = stroke[:3], stroke[3:6]
        

        # Update min and max limits based on strokes (ignoring circles)
        if stroke[-1] == 1:
            # straight line
            x_min, x_max = min(x_min, start[0], end[0]), max(x_max, start[0], end[0])
            y_min, y_max = min(y_min, start[1], end[1]), max(y_max, start[1], end[1])
            z_min, z_max = min(z_min, start[2], end[2]), max(z_max, start[2], end[2])
        
        if stroke[-1] == 2:
            # Circle face
            x_values, y_values, z_values = plot_circle(stroke)
            ax.plot(x_values, y_values, z_values, color='red', alpha=1, linewidth=0.5)
            continue

        if stroke[-1] ==3:
            # Arc
            x_values, y_values, z_values = plot_arc(stroke)
            ax.plot(x_values, y_values, z_values, color='blue', alpha=1, linewidth=0.5)
            continue

        else:
            # Hand-drawn effect for regular stroke line
            x_values = np.array([start[0], end[0]])
            y_values = np.array([start[1], end[1]])
            z_values = np.array([start[2], end[2]])
            
            # Add perturbations for hand-drawn effect
            perturbations = np.random.normal(0, perturb_factor, (10, 3))
            t = np.linspace(0, 1, 10)
            x_interpolated = np.linspace(x_values[0], x_values[1], 10) + perturbations[:, 0]
            y_interpolated = np.linspace(y_values[0], y_values[1], 10) + perturbations[:, 1]
            z_interpolated = np.linspace(z_values[0], z_values[1], 10) + perturbations[:, 2]

            # Smooth curve with cubic splines
            cs_x = CubicSpline(t, x_interpolated)
            cs_y = CubicSpline(t, y_interpolated)
            cs_z = CubicSpline(t, z_interpolated)
            smooth_t = np.linspace(0, 1, 100)
            smooth_x = cs_x(smooth_t)
            smooth_y = cs_y(smooth_t)
            smooth_z = cs_z(smooth_t)

            # Plot perturbed line
            ax.plot(smooth_x, smooth_y, smooth_z, color='black', alpha=1, linewidth=0.5)


    # Compute the center and rescale
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    z_center = (z_min + z_max) / 2
    max_diff = max(x_max - x_min, y_max - y_min, z_max - z_min)
    ax.set_xlim([x_center - max_diff / 2, x_center + max_diff / 2])
    ax.set_ylim([y_center - max_diff / 2, y_center + max_diff / 2])
    ax.set_zlim([z_center - max_diff / 2, z_center + max_diff / 2])

    # Remove axis ticks and labels
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    # Show plot
    plt.show()




def vis_stroke_node_features_and_highlights(stroke_node_features, added_feature_lines):
    # Initialize the 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.grid(False)
    ax.axis('off')  # Turn off axis background and borders

    # Initialize min and max limits
    x_min, x_max = float('inf'), float('-inf')
    y_min, y_max = float('inf'), float('-inf')
    z_min, z_max = float('inf'), float('-inf')

    perturb_factor = 0.000002  # Adjusted perturbation factor for hand-drawn effect

    # Plot all strokes in blue with perturbations
    for idx, stroke in enumerate(stroke_node_features):
        start, end = stroke[:3], stroke[3:6]

        # Update min and max limits based on strokes (ignoring circles)
        if stroke[-1] == 1:
            # straight line
            x_min, x_max = min(x_min, start[0], end[0]), max(x_max, start[0], end[0])
            y_min, y_max = min(y_min, start[1], end[1]), max(y_max, start[1], end[1])
            z_min, z_max = min(z_min, start[2], end[2]), max(z_max, start[2], end[2])

        if stroke[-1] == 2:
            # Circle face
            x_values, y_values, z_values = plot_circle(stroke)
            ax.plot(x_values, y_values, z_values, color='black', alpha=1, linewidth=0.5)
            continue

        if stroke[-1] == 3:
            # Arc
            x_values, y_values, z_values = plot_arc(stroke)
            ax.plot(x_values, y_values, z_values, color='black', alpha=1, linewidth=0.5)
            continue

        else:
            # Hand-drawn effect for regular stroke line
            x_values = np.array([start[0], end[0]])
            y_values = np.array([start[1], end[1]])
            z_values = np.array([start[2], end[2]])

            # Add perturbations for hand-drawn effect
            perturbations = np.random.normal(0, perturb_factor, (10, 3))
            t = np.linspace(0, 1, 10)
            x_interpolated = np.linspace(x_values[0], x_values[1], 10) + perturbations[:, 0]
            y_interpolated = np.linspace(y_values[0], y_values[1], 10) + perturbations[:, 1]
            z_interpolated = np.linspace(z_values[0], z_values[1], 10) + perturbations[:, 2]

            # Smooth curve with cubic splines
            cs_x = CubicSpline(t, x_interpolated)
            cs_y = CubicSpline(t, y_interpolated)
            cs_z = CubicSpline(t, z_interpolated)
            smooth_t = np.linspace(0, 1, 100)
            smooth_x = cs_x(smooth_t)
            smooth_y = cs_y(smooth_t)
            smooth_z = cs_z(smooth_t)

            # Plot perturbed line
            ax.plot(smooth_x, smooth_y, smooth_z, color='black', alpha=1, linewidth=0.5)

    # Plot added feature lines in red
    for line in added_feature_lines:
        start, end = line[:3], line[3:6]
        ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]],
                color='red', linewidth=1.0, alpha=1)

    # Compute the center and rescale
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    z_center = (z_min + z_max) / 2
    max_diff = max(x_max - x_min, y_max - y_min, z_max - z_min)
    ax.set_xlim([x_center - max_diff / 2, x_center + max_diff / 2])
    ax.set_ylim([y_center - max_diff / 2, y_center + max_diff / 2])
    ax.set_zlim([z_center - max_diff / 2, z_center + max_diff / 2])

    # Remove axis ticks and labels
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    # Show plot
    plt.show()



def vis_points(unique_points):
    """
    Visualize only the 3D points in space without axes, background, or automatic zooming.

    Parameters:
    - unique_points (list or set): Collection of 3D points (tuples or lists).
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    # Initialize the 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Remove axis labels, ticks, and background
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_frame_on(False)
    ax.grid(False)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.set_axis_off()  # Hides the axis completely

    # Initialize bounding box variables
    x_min, x_max = float('inf'), float('-inf')
    y_min, y_max = float('inf'), float('-inf')
    z_min, z_max = float('inf'), float('-inf')

    # Extract coordinates
    for pt in unique_points:
        x, y, z = pt[0], pt[1], pt[2]

        # Update bounding box
        x_min, x_max = min(x_min, x), max(x_max, x)
        y_min, y_max = min(y_min, y), max(y_max, y)
        z_min, z_max = min(z_min, z), max(z_max, z)

        # Plot the point
        ax.scatter(x, y, z, color='black', s=4)

    # Compute the center and rescale
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    z_center = (z_min + z_max) / 2
    max_diff = max(x_max - x_min, y_max - y_min, z_max - z_min)

    ax.set_xlim([x_center - max_diff / 2, x_center + max_diff / 2])
    ax.set_ylim([y_center - max_diff / 2, y_center + max_diff / 2])
    ax.set_zlim([z_center - max_diff / 2, z_center + max_diff / 2])

    # Show the plot
    plt.show()
 


def vis_brep(brep):
    """
    Visualize the brep strokes and circular/cylindrical faces in 3D space if brep is not empty.
    
    Parameters:
    brep (np.ndarray or torch.Tensor): A matrix with shape (num_strokes, 6) representing strokes.
                       Each row contains two 3D points representing the start and end of a stroke.
                       If brep.shape[0] == 0, the function returns without plotting.
    """
    # Initialize the 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.grid(False)

    # Check if brep is empty
    if brep.shape[0] == 0:
        plt.title('Empty Plot')
        plt.show()
        return

    # Convert brep to numpy if it's a tensor
    if not isinstance(brep, np.ndarray):
        brep = brep.numpy()

    # Initialize min and max limits
    x_min, x_max = float('inf'), float('-inf')
    y_min, y_max = float('inf'), float('-inf')
    z_min, z_max = float('inf'), float('-inf')

    # Plot all brep strokes and circle/cylinder faces in blue

    # Last values
    # Straight Line: 1
    # Circle Feature: 2
    # Cylinder Face Feature: 3
    # Arc Feature: 4
    for stroke in brep:
        
        if stroke[-1] == 3:

            # Cylinder face
            center = stroke[:3]
            normal = stroke[3:6]
            height = stroke[6]
            radius = stroke[7]

            # Generate points for the cylinder's base circle (less dense)
            theta = np.linspace(0, 2 * np.pi, 30)  # Less dense with 30 points
            x_values = radius * np.cos(theta)
            y_values = radius * np.sin(theta)
            z_values = np.zeros_like(theta)

            # Combine the coordinates into a matrix (3, 30)
            base_circle_points = np.array([x_values, y_values, z_values])

            # Normalize the normal vector
            normal = normal / np.linalg.norm(normal)

            # Rotation logic using Rodrigues' formula
            z_axis = np.array([0, 0, 1])  # Z-axis is the default normal for the cylinder

            # Rotate the base circle points to align with the normal vector (even if normal is aligned)
            rotation_axis = np.cross(z_axis, normal)
            if np.linalg.norm(rotation_axis) > 0:  # Check if rotation is needed
                rotation_axis /= np.linalg.norm(rotation_axis)
                angle = np.arccos(np.clip(np.dot(z_axis, normal), -1.0, 1.0))

                # Create the rotation matrix using the rotation axis and angle (Rodrigues' rotation formula)
                K = np.array([[0, -rotation_axis[2], rotation_axis[1]],
                              [rotation_axis[2], 0, -rotation_axis[0]],
                              [-rotation_axis[1], rotation_axis[0], 0]])

                R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)

                # Rotate the base circle points
                rotated_base_circle_points = np.dot(R, base_circle_points)
            else:
                rotated_base_circle_points = base_circle_points

            # Translate the base circle to the center point
            x_base = rotated_base_circle_points[0] + center[0]
            y_base = rotated_base_circle_points[1] + center[1]
            z_base = rotated_base_circle_points[2] + center[2]

            # Plot the base circle
            ax.plot(x_base, y_base, z_base, color='blue')

            # Plot vertical lines to create the "cylinder" (but without filling the body)
            x_top = x_base - normal[0] * height
            y_top = y_base - normal[1] * height
            z_top = z_base - normal[2] * height

            # Plot lines connecting the base and top circle with reduced density
            for i in range(0, len(x_base), 3):  # Fewer lines by skipping points
                ax.plot([x_base[i], x_top[i]], [y_base[i], y_top[i]], [z_base[i], z_top[i]], color='blue')


        elif stroke[-1] == 2:
            # Circle face (same rotation logic as shared)
            x_values, y_values, z_values = plot_circle(stroke)
            ax.plot(x_values, y_values, z_values, color='blue')
        
        elif stroke[-1] == 4:
            # plot arc 
            x_values, y_values, z_values = plot_arc(stroke)
            ax.plot(x_values, y_values, z_values, color='blue')


        else:
            # Plot the stroke
            start, end = stroke[:3], stroke[3:6]
            ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], color='blue', linewidth=1)

            # Update axis limits for the stroke points
            x_min, x_max = min(x_min, start[0], end[0]), max(x_max, start[0], end[0])
            y_min, y_max = min(y_min, start[1], end[1]), max(y_max, start[1], end[1])
            z_min, z_max = min(z_min, start[2], end[2]), max(z_max, start[2], end[2])

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



def plot_circle(stroke):
    center = stroke[:3]
    normal = stroke[3:6]
    radius = stroke[7]

    # Generate circle points in the XY plane
    theta = np.linspace(0, 2 * np.pi, 30)  # Less dense with 30 points
    x_values = radius * np.cos(theta)
    y_values = radius * np.sin(theta)
    z_values = np.zeros_like(theta)

    # Combine the coordinates into a matrix (3, 30)
    circle_points = np.array([x_values, y_values, z_values])

    # Normalize the normal vector
    normal = normal / np.linalg.norm(normal)

    # Rotation logic using Rodrigues' formula
    z_axis = np.array([0, 0, 1])  # Z-axis is the default normal for the circle

    rotation_axis = np.cross(z_axis, normal)
    if np.linalg.norm(rotation_axis) > 0:  # Check if rotation is needed
        rotation_axis /= np.linalg.norm(rotation_axis)
        angle = np.arccos(np.clip(np.dot(z_axis, normal), -1.0, 1.0))

        # Create the rotation matrix using the rotation axis and angle (Rodrigues' rotation formula)
        K = np.array([[0, -rotation_axis[2], rotation_axis[1]],
                        [rotation_axis[2], 0, -rotation_axis[0]],
                        [-rotation_axis[1], rotation_axis[0], 0]])

        R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)

        # Rotate the circle points
        rotated_circle_points = np.dot(R, circle_points)
    else:
        rotated_circle_points = circle_points

    # Translate the circle to the center point
    x_values = rotated_circle_points[0] + center[0]
    y_values = rotated_circle_points[1] + center[1]
    z_values = rotated_circle_points[2] + center[2]


    return x_values, y_values, z_values



def plot_arc(stroke):
    import numpy as np

    # Extract start and end points from the stroke
    start_point = np.array(stroke[:3])
    end_point = np.array(stroke[3:6])

    # Generate a straight line with 100 points between start_point and end_point
    t = np.linspace(0, 1, 100)  # Parameter for interpolation
    line_points = (1 - t)[:, None] * start_point + t[:, None] * end_point

    # Return x, y, z coordinates of the line points
    return line_points[:, 0], line_points[:, 1], line_points[:, 2]


# ------------------------------------------------------------------------------------# 


def extract_feature_lines(final_edges_data):
    """
    Extracts strokes from final_edges_data where type is 'feature_line'.

    Parameters:
    - final_edges_data (dict): A dictionary where keys are stroke IDs and values contain stroke properties.

    Returns:
    - list: A list of strokes that are labeled as 'feature_line'.
    """
    feature_lines = []

    for key, stroke in final_edges_data.items():
        stroke_type = stroke['type']

        if stroke_type == 'feature_line' or stroke_type == 'extrude_line' or stroke_type == 'fillet_line':
            feature_lines.append(stroke)

    return feature_lines


def extract_all_lines(final_edges_data):
    feature_lines = []

    for key, stroke in final_edges_data.items():
        
        feature_lines.append(stroke)

    return feature_lines


def extract_only_construction_lines(final_edges_data):
    """
    Extracts strokes from final_edges_data where type is 'feature_line'.

    Parameters:
    - final_edges_data (dict): A dictionary where keys are stroke IDs and values contain stroke properties.

    Returns:
    - list: A list of strokes that are labeled as 'feature_line'.
    """
    feature_lines = []

    for key, stroke in final_edges_data.items():
        stroke_type = stroke['type']
        
        if stroke_type != 'feature_line' and stroke_type != 'extrude_line' and stroke_type != 'fillet_line':
            feature_lines.append(stroke)

    return feature_lines



def point_on_line_extension(p, a, b, tol=1e-7):
    """
    Check if point p lies on the infinite line defined by points a and b.
    """
    a, b, p = np.array(a), np.array(b), np.array(p)
    ab = b - a
    ap = p - a
    cross = np.cross(ab, ap)
    return np.linalg.norm(cross) < tol


def split_and_merge_stroke_cloud(stroke_node_features, is_feature_line_matrix):
    stroke_node_features = np.array(stroke_node_features)
    add_feature_lines = []

    # Step 1: Gather unique points from feature lines
    unique_points = set()
    for i, stroke in enumerate(stroke_node_features):
        if stroke[-1] == 1 and is_feature_line_matrix[i]:
            point_1 = tuple(np.round(stroke[:3], 4))
            point_2 = tuple(np.round(stroke[3:6], 4))
            unique_points.add(point_1)
            unique_points.add(point_2)

    unique_points = list(unique_points)


    # Step 2.1: Find sets of collinear points
    collinear_candidate_sets = []

    for i, stroke in enumerate(stroke_node_features):
        if stroke[-1] != 1 or not is_feature_line_matrix[i]:
            continue
        p1 = tuple(np.round(stroke[:3], 5))
        p2 = tuple(np.round(stroke[3:6], 5))
        collinear_candidate_sets.append(set([p1, p2]))

    # Step 2.2: Enrich each set with all other points that lie on the same line
    for point in unique_points:
        for s in collinear_candidate_sets:
            s_list = list(s)
            if len(s_list) >= 2:
                p1, p2 = s_list[0], s_list[1]
                if point != p1 and point != p2 and point_on_line_extension(point, p1, p2):
                    s.add(point)

    # Step 2.3: Sort and deduplicate sets
    normalized_sets = []
    seen_sets = set()

    for s in collinear_candidate_sets:
        if len(s) > 2:
            sorted_tuple = tuple(sorted(s))
            if sorted_tuple not in seen_sets:
                seen_sets.add(sorted_tuple)
                normalized_sets.append(sorted_tuple)

    # Step 2.4: Final result
    collinear_sets = normalized_sets

    # for c_set in collinear_candidate_sets:
    #     print("set", c_set)

    # Step 3: From each collinear set, generate all segments (in order of distance)
    for col_set in collinear_sets:
        col_set_np = np.array(col_set)
        # Sort by distance along the first axis
        anchor = col_set_np[0]
        sorted_points = sorted(col_set_np, key=lambda x: np.linalg.norm(x - anchor))
        for i in range(len(sorted_points) - 1):
            pt1 = sorted_points[i]
            pt2 = sorted_points[i + 1]
            new_line = np.concatenate([pt1, pt2])
            # Check if line already exists
            exists = False
            for stroke in stroke_node_features:
                s = np.round(stroke[:3], 4)
                e = np.round(stroke[3:6], 4)
                if (np.allclose(s, pt1) and np.allclose(e, pt2)) or \
                   (np.allclose(s, pt2) and np.allclose(e, pt1)):
                    exists = True
                    break
            if not exists:
                # Add feature stroke
                new_line = list(new_line) + [0] + [0, 0, 0, 1]
                add_feature_lines.append(new_line)
        
    updated_strokes = np.concatenate([stroke_node_features, np.array(add_feature_lines)], axis=0) \
        if add_feature_lines else stroke_node_features

    return updated_strokes, np.array(add_feature_lines)

import numpy as np

def are_colinear(p1, p2, p3):
    v1 = np.array(p2) - np.array(p1)
    v2 = np.array(p3) - np.array(p1)
    cross = np.cross(v1, v2)
    return np.allclose(cross, [0, 0, 0], atol=1e-5)

def points_are_close(pt1, pt2, tol=1e-5):
    return np.allclose(pt1, pt2, atol=tol)

def contained(edge1, edge2):
    p1, p2 = np.array(edge1[:3]), np.array(edge1[3:6])
    q1, q2 = np.array(edge2[:3]), np.array(edge2[3:6])
    
    if not are_colinear(p1, p2, q1) or not are_colinear(p1, p2, q2):
        return False

    vec = p2 - p1
    vec_len = np.linalg.norm(vec)
    if vec_len < 1e-8:
        return False
    direction = vec / vec_len

    def proj_fraction(point): return np.dot(point - p1, direction) / vec_len
    f1, f2 = proj_fraction(q1), proj_fraction(q2)

    if 0 - 1e-5 <= f1 <= 1 + 1e-5 and 0 - 1e-5 <= f2 <= 1 + 1e-5:
        return True

    # Check reverse containment
    vec2 = q2 - q1
    len2 = np.linalg.norm(vec2)
    if len2 < 1e-8:
        return False
    dir2 = vec2 / len2

    def proj2(p): return np.dot(p - q1, dir2) / len2
    r1, r2 = proj2(p1), proj2(p2)

    if 0 - 1e-5 <= r1 <= 1 + 1e-5 and 0 - 1e-5 <= r2 <= 1 + 1e-5:
        return True

    return False

def merge_edges(edge1, edge2):
    points = [edge1[:3], edge1[3:6], edge2[:3], edge2[3:6]]

    # Count occurrences with fuzzy comparison
    point_counts = []
    for i, pt in enumerate(points):
        count = 0
        for other in points:
            if points_are_close(pt, other):
                count += 1
        point_counts.append((pt, count))

    # Keep points that appear only once
    unique_points = [pt for pt, count in point_counts if count == 1]


    if len(unique_points) != 2:
        # fallback: pick two most distant points
        max_dist = -1
        pt1, pt2 = points[0], points[1]
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                dist = np.linalg.norm(np.array(points[i]) - np.array(points[j]))
                if dist > max_dist:
                    max_dist = dist
                    pt1, pt2 = points[i], points[j]
        unique_points = [pt1, pt2]

    return list(unique_points[0]) + list(unique_points[1]) + [0, 0, 0, 1]


def only_merge_brep(edge_features_list):
    def _run_single_pass(edges):
        new_features_list = []
        used_indices = set()

        for i, edge1 in enumerate(edges):
            if i in used_indices:
                continue
            if edge1[-1] != 1:
                new_features_list.append(edge1)
                continue

            merged = False
            for j, edge2 in enumerate(edges):
                if i == j or j in used_indices or edge2[-1] != 1:
                    continue

                p11, p12 = edge1[:3], edge1[3:6]
                p21, p22 = edge2[:3], edge2[3:6]

                common_found = any(points_are_close(pa, pb) for pa in [p11, p12] for pb in [p21, p22])
                if not common_found:
                    continue

                all_points = [p11, p12, p21, p22]
                unique_points = []
                for pt in all_points:
                    if not any(points_are_close(pt, up) for up in unique_points):
                        unique_points.append(pt)

                if len(unique_points) < 3:
                    continue

                if are_colinear(unique_points[0], unique_points[1], unique_points[2]) and not contained(edge1, edge2):
                    merged_edge = merge_edges(edge1, edge2)
                    new_features_list.append(merged_edge)
                    used_indices.update([i, j])
                    merged = True
                    break

            if not merged:
                new_features_list.append(edge1)

        return new_features_list

    prev_length = -1
    current_edges = edge_features_list

    while True:
        merged_edges = _run_single_pass(current_edges)
        if len(merged_edges) == prev_length:
            break
        prev_length = len(merged_edges)
        current_edges = merged_edges

    return current_edges




def split_and_merge_brep(edge_features_list):
    add_feature_lines = []

    # Step 1: Gather unique points from feature lines
    unique_points = set()
    for i, stroke in enumerate(edge_features_list):
        if stroke[-1] == 1:
            point_1 = tuple(np.round(stroke[:3], 5))
            point_2 = tuple(np.round(stroke[3:6], 5))
            unique_points.add(point_1)
            unique_points.add(point_2)

    unique_points = list(unique_points)


    # Step 2.1: Find sets of collinear points
    collinear_candidate_sets = []

    for i, stroke in enumerate(edge_features_list):
        if stroke[-1] != 1:
            continue
        p1 = tuple(np.round(stroke[:3], 5))
        p2 = tuple(np.round(stroke[3:6], 5))
        collinear_candidate_sets.append(set([p1, p2]))

    # Step 2.2: Enrich each set with all other points that lie on the same line
    for point in unique_points:
        for s in collinear_candidate_sets:
            s_list = list(s)
            if len(s_list) >= 2:
                p1, p2 = s_list[0], s_list[1]
                if point != p1 and point != p2 and point_on_line_extension(point, p1, p2):
                    s.add(point)

    # Step 2.3: Sort and deduplicate sets
    normalized_sets = []
    seen_sets = set()

    for s in collinear_candidate_sets:
        if len(s) > 2:
            sorted_tuple = tuple(sorted(s))
            if sorted_tuple not in seen_sets:
                seen_sets.add(sorted_tuple)
                normalized_sets.append(sorted_tuple)

    # Step 2.4: Final result
    collinear_sets = normalized_sets

    # for c_set in collinear_sets:
    #     print("set", c_set)

    # Step 3: From each collinear set, generate all segments (in order of distance)
    for col_set in collinear_sets:
        col_set_np = np.array(col_set)
        sorted_points = sorted(col_set_np, key=lambda x: tuple(x))  # consistent ordering
        for pt1, pt2 in combinations(sorted_points, 2):  # all point pairs
            exists = False
            for edge in edge_features_list:
                s = np.round(edge[:3], 4)
                e = np.round(edge[3:6], 4)
                if (np.allclose(s, pt1) and np.allclose(e, pt2)) or \
                (np.allclose(s, pt2) and np.allclose(e, pt1)):
                    exists = True
                    break
            if not exists:
                new_line = np.concatenate([pt1, pt2])
                new_line = list(new_line) + [0, 0, 0, 1]  # dummy metadata
                add_feature_lines.append(new_line)
    
    return edge_features_list + add_feature_lines, add_feature_lines



def ensure_loop(stroke_node_features, selected_indices, tol=1e-4):
    """
    Checks whether the selected strokes form a valid loop:
    All endpoints (3D points) must appear exactly twice (within distance tolerance).

    Parameters:
    - stroke_node_features: np.ndarray of shape (num_strokes, ≥6)
    - selected_indices: list of indices (ints)
    - tol: distance tolerance for point matching

    Returns:
    - True if all points appear exactly twice, False otherwise
    """
    points = []

    # Collect all points (start and end) from selected strokes
    for idx in selected_indices:
        stroke = stroke_node_features[idx]
        points.append(stroke[0:3])
        points.append(stroke[3:6])

    # Count occurrences using distance-based matching
    matched_flags = [False] * len(points)
    counts = [0] * len(points)

    for i, p1 in enumerate(points):
        if matched_flags[i]:
            continue
        for j, p2 in enumerate(points):
            if np.linalg.norm(np.array(p1) - np.array(p2)) < tol:
                counts[i] += 1
                matched_flags[j] = True  # mark as used in match

    # Each point must appear exactly twice
    for c in counts:
        if c==2 or c==0:
            return True

    return False

# ------------------------------------------------------------------------------------# 
def extract_input_json(final_edges_data, strokes_dict_data, subfolder_path):
    """
    Extracts stroke data from final_edges_data and saves it as 'input.json' in the specified subfolder.

    Parameters:
    - final_edges_data: Dictionary containing stroke information.
    - subfolder_path: Path where the JSON file should be saved.
    """
    strokes = []
    stroke_id_mapping = {}  # Maps stroke keys to index IDs
    current_id = 0

    for key, stroke in final_edges_data.items():
        stroke_type = stroke["type"]

        # Only consider feature, extrude, and fillet lines
        if stroke_type in ["feature_line", "extrude_line", "fillet_line"]:
            geometry = stroke["geometry"]

            if len(geometry) == 2:
                # Straight line: (x1, y1, z1, x2, y2, z2)
                stroke_data = {
                    "id": current_id,
                    "type": "line",
                    "coords": [*geometry[0], *geometry[1]]  # Flatten start & end points
                }
            else:
                # Curve line: (x1, y1, z1, x2, y2, z2, cx, cy, cz)
                start = geometry[0]
                end = geometry[-1]
                control = geometry[1]  # Assuming single control point for now

                stroke_data = {
                    "id": current_id,
                    "type": "curve",
                    "coords": [*start, *end, *control]
                }

            strokes.append(stroke_data)
            stroke_id_mapping[key] = current_id
            current_id += 1

    # Extract intersections based on geometry proximity (to be implemented)
    intersections = extract_intersections(strokes_dict_data)

    dataset_entry = {
        "strokes": strokes,
        "intersections": intersections,
        "construction_lines": []  # Placeholder until we define a method
    }

    # Ensure the folder exists before saving
    if not os.path.exists(subfolder_path):
        os.makedirs(subfolder_path)
        
    json_path = os.path.join(subfolder_path, "input.json")

    # Save to file
    with open(json_path, "w") as f:
        json.dump(dataset_entry, f, indent=4)



def extract_intersections(strokes_dict_data):
    intersections = []

    for idx, stroke_dict in enumerate(strokes_dict_data):
        intersect_strokes = stroke_dict["intersections"]

        # Unfold the sublists to get all intersecting stroke indices
        intersecting_indices = {stroke_idx for sublist in intersect_strokes for stroke_idx in sublist}

        # Add intersections as pairs (ensuring stroke_1 < stroke_2 for consistency)
        for intersecting_idx in intersecting_indices:
            if 0 <= intersecting_idx < len(strokes_dict_data):  # Ensure index is valid
                intersection_pair = tuple(sorted([idx, intersecting_idx]))  # Ensure order consistency
                if intersection_pair not in intersections:
                    intersections.append(intersection_pair)

    return intersections




# ------------------------------------------------------------------------------------# 
def compute_midpoint(stroke):
    """Compute the midpoint of a feature stroke."""
    start, end = stroke['geometry'][0], stroke['geometry'][-1]
    return [(start[0] + end[0]) / 2, (start[1] + end[1]) / 2, (start[2] + end[2]) / 2]

def is_close(p1, p2, tol=1e-3):
    """Check if two points are approximately the same within a given tolerance."""
    return all(abs(a - b) < tol for a, b in zip(p1, p2))

def point_meaning(point, feature_lines):
    """
    Determine the meaning of a given point relative to feature strokes.

    Parameters:
    - point: A 3D point [x, y, z]
    - feature_lines: A list of feature strokes as dictionaries {id, geometry}

    Returns:
    - A tuple (relation, feature_line_id) or ("unknown", -1) if no relation found.
    """
    for stroke in feature_lines:
        stroke_id = stroke['id']
        start, end = stroke['geometry'][0], stroke['geometry'][-1]
        midpoint = compute_midpoint(stroke)

        if is_close(point, start):
            return ("endpoint", stroke_id)
        elif is_close(point, end):
            return ("endpoint", stroke_id)
        elif is_close(point, midpoint):
            return ("midpoint", stroke_id)

    # Check if the point lies on an extension of any feature stroke
    for stroke in feature_lines:
        stroke_id = stroke['id']
        start, end = stroke['geometry'][0], stroke['geometry'][-1]
        stroke_vec = [end[i] - start[i] for i in range(3)]
        point_vec = [point[i] - start[i] for i in range(3)]

        # Check collinearity using cross product
        cross_product = [
            stroke_vec[1] * point_vec[2] - stroke_vec[2] * point_vec[1],
            stroke_vec[2] * point_vec[0] - stroke_vec[0] * point_vec[2],
            stroke_vec[0] * point_vec[1] - stroke_vec[1] * point_vec[0]
        ]

        if all(abs(c) < 1e-3 for c in cross_product):  # Collinear check
            dot_product = sum(stroke_vec[i] * point_vec[i] for i in range(3))
            stroke_length = sum(stroke_vec[i] ** 2 for i in range(3)) ** 0.5
            point_length = sum(point_vec[i] ** 2 for i in range(3)) ** 0.5

            if dot_product > 0 and point_length > stroke_length:
                return ("on_extension", stroke_id)

    return ("unknown", -1)

def assign_point_meanings(construction_lines, feature_lines, subfolder_path):
    """
    Assign meanings to the two endpoints of each construction line and save them as gt_output.json.

    Parameters:
    - construction_lines: List of construction lines as dictionaries {id, geometry}
    - feature_lines: List of feature strokes as dictionaries {id, geometry}
    - subfolder_path: Path where the JSON file should be saved.

    Returns:
    - Saves the output JSON file containing labels.
    """
    labeled_data = []

    for construction in construction_lines:
        point1, point2 = construction['geometry'][0], construction['geometry'][-1]

        meaning1 = point_meaning(point1, feature_lines)
        meaning2 = point_meaning(point2, feature_lines)

        labeled_data.append([meaning1, meaning2])

    # Ensure the folder exists before saving
    if not os.path.exists(subfolder_path):
        os.makedirs(subfolder_path)

    json_path = os.path.join(subfolder_path, "gt_output.json")

    # Save to file
    with open(json_path, "w") as f:
        json.dump(labeled_data, f, indent=4)

    