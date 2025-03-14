import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
from scipy.optimize import least_squares
from scipy.interpolate import splprep, splev, CubicSpline


from itertools import product


import json
import os




# ------------------------------------------------------------------------------------# 



def build_final_edges_json(final_edges_json):
    node_features_list = []

    for key in final_edges_json.keys():
        stroke = final_edges_json[key]

        geometry = stroke["geometry"]

        node_feature = build_node_features(geometry)

        node_features_list.append(node_feature)

    node_features_matrix = np.array(node_features_list)

    return node_features_matrix



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

    # Case 1: Check if the geometry has low residual fitting straight line  -> (Straight Line)
    residual = fit_straight_line(geometry)
    threshold = dist(geometry) * 2

    if residual < threshold:
        point1 = geometry[0]
        point2 = geometry[1]

        return point1 + point2 + [0, 0, 0, 1]

    # Check if geometry is closed
    distance, closed = is_closed_shape(geometry)

    if not closed or len(geometry) < 5:
        center_circle, radius_circle, normal_circle, circle_residual = fit_circle_3d(geometry)

        if circle_residual < threshold:
            # Case 3: Arc
            point1 = geometry[0]
            point2 = geometry[-1]
            return point1 + point2 + center_circle + [3]

        # Case 6: Curved Line
        point1 = geometry[0]
        point2 = geometry[-1]
        random_point = geometry[len(geometry) // 2]
        return point1 + point2 + random_point + [6]

    # Try fitting a circle
    center_circle, radius_circle, normal_circle, circle_residual = fit_circle_3d(geometry)

    if circle_residual > threshold:
        # Case 5: Closed Shape
        point1 = geometry[0]
        point2 = geometry[-1]
        random_point = geometry[len(geometry) // 2]

        return point1 + point2 + random_point + [5]



    # Try fitting an ellipse
    center_ellipse, normal_ellipse, axes_lengths, theta, ellipse_residual = fit_ellipse_3d(geometry)
    major_axis, minor_axis = axes_lengths

    if abs(major_axis - minor_axis) < threshold:
        # Case 4: Ellipse
        return center_ellipse + normal_ellipse+ [major_axis, minor_axis, 0, 4]
    
    # Case 2: Circle
    return center_circle + normal_circle + [radius_circle, 0, 0, 2]



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

    # print("points", points[0], points[-1])
    # print("center_init", center_init)
    # print("center_opt:", center_opt)
    # print("radius_opt", radius_opt)
    # print("normal:", normal)
    # print("np.mean(np.abs(final_residuals))", np.mean(np.abs(final_residuals)))
    # print("-------")

    return list(center_opt), radius_opt, list(normal), np.mean(np.abs(final_residuals))



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
