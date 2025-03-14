import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
from scipy.optimize import least_squares
from scipy.interpolate import splprep, splev, CubicSpline


from itertools import product


import json
import os



# ------------------------------------------------------------------------------------# 


def vis_feature_lines(feature_lines):
    """
    Visualize only the feature_line strokes in 3D space.

    Parameters:
    - feature_lines (list): List of stroke dictionaries containing geometry (list of 3D points).
    """
    # Initialize the 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.grid(False)

    # Loop through all feature_line strokes
    for stroke in feature_lines:
        geometry = stroke["geometry"]

        if len(geometry) < 2:
            continue  # Ensure there are enough points to plot

        # Plot each segment of the stroke
        for j in range(1, len(geometry)):
            start = geometry[j - 1]
            end = geometry[j]

            # Extract coordinates for plotting
            x_values = [start[0], end[0]]
            y_values = [start[1], end[1]]
            z_values = [start[2], end[2]]

            # Plot the stroke as a black line
            ax.plot(x_values, y_values, z_values, color='black', linewidth=0.5)

    # Set axis labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

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
