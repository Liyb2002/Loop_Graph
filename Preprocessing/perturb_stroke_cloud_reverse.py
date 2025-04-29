import random
import numpy as np

import random

def stroke_node_features_to_polyline(stroke_node_features, is_feature_line):
    """
    Convert stroke_node_features into a list of stroke objects (polyline format).

    Each stroke object has:
    - type: "feature_line" or "normal_line"
    - feature_id: 0 (for now)
    - geometry: list of 3D points
    - id: unique id per stroke
    - opacity: random in [0.6, 0.8] for feature lines, [0.1, 0.3] otherwise
    """
    stroke_cloud = build_stroke_cloud_from_node_features(stroke_node_features)

    stroke_objects = []

    for idx, pts in enumerate(stroke_cloud):
        if is_feature_line[idx][0] == 1:
            opacity = random.uniform(0.6, 0.8)
            stroke_type = "feature_line"
        else:
            opacity = random.uniform(0.1, 0.3)
            stroke_type = "normal_line"

        stroke_obj = {
            "type": stroke_type,
            "feature_id": 0,
            "geometry": pts.tolist(),
            "id": idx,
            "opacity": opacity
        }
        stroke_objects.append(stroke_obj)

    return stroke_objects


def build_stroke_cloud_from_node_features(stroke_node_features, 
                                           num_points_straight=5, 
                                           num_points_arc=10, 
                                           num_points_circle=20):
    """
    Build a perturbed stroke cloud from clean stroke_node_features.

    Returns:
    - List of (M, 3) arrays, one array per perturbed stroke
    """
    stroke_cloud = []

    for stroke in stroke_node_features:
        if all(v == 0 for v in stroke):
            continue  # skip padded strokes

        stroke_type = stroke[-1]

        if stroke_type == 1:  # Straight line
            start = stroke[0:3]
            end = stroke[3:6]
            pts = np.linspace(start, end, num_points_straight)

            stroke_cloud.append(pts)

        elif stroke_type == 3:  # Arc
            pts = reconstruct_arc_points(stroke, num_points=num_points_arc)
            stroke_cloud.append(pts)

        elif stroke_type == 2:  # Circle
            pts = reconstruct_circle_points(stroke, num_points=num_points_circle)
            stroke_cloud.append(pts)

    return stroke_cloud



import numpy as np

def reconstruct_arc_points(stroke, num_points=10):
    """
    Reconstruct a 1/4 circle arc based on center, start, and end points.
    The arc is axis-aligned (XY, YZ, or XZ plane).
    """
    start = np.array(stroke[0:3])
    end = np.array(stroke[3:6])
    center = np.array(stroke[7:10])
    
    # Determine the plane
    tol = 1e-4
    if abs(start[0] - center[0]) < tol and abs(end[0] - center[0]) < tol:
        plane = 'YZ'
        idx1, idx2 = 1, 2
        fixed_idx = 0
    elif abs(start[1] - center[1]) < tol and abs(end[1] - center[1]) < tol:
        plane = 'XZ'
        idx1, idx2 = 0, 2
        fixed_idx = 1
    elif abs(start[2] - center[2]) < tol and abs(end[2] - center[2]) < tol:
        plane = 'XY'
        idx1, idx2 = 0, 1
        fixed_idx = 2
    else:
        raise ValueError("Arc is not axis-aligned.")

    # Project to 2D plane
    start_2d = np.array([start[idx1], start[idx2]])
    center_2d = np.array([center[idx1], center[idx2]])

    radius = np.linalg.norm(start_2d - center_2d)

    # Compute starting angle
    start_angle = np.arctan2(start_2d[1] - center_2d[1], start_2d[0] - center_2d[0])

    # Since it's a 1/4 circle, the end angle is start + 90 degrees
    # But we have to decide CW or CCW depending on end point

    # Predict two possibilities
    ccw_end_angle = start_angle + (np.pi / 2)   # +90 degree
    cw_end_angle  = start_angle - (np.pi / 2)   # -90 degree

    # Project end point
    end_2d = np.array([end[idx1], end[idx2]])
    target_vec = end_2d - center_2d

    # Compare which angle (ccw or cw) is closer to the actual end
    ccw_vec = np.array([np.cos(ccw_end_angle), np.sin(ccw_end_angle)])
    cw_vec = np.array([np.cos(cw_end_angle), np.sin(cw_end_angle)])

    if np.linalg.norm(target_vec/np.linalg.norm(target_vec) - ccw_vec) < np.linalg.norm(target_vec/np.linalg.norm(target_vec) - cw_vec):
        chosen_end_angle = ccw_end_angle
    else:
        chosen_end_angle = cw_end_angle

    # Now sample angles between start and chosen_end_angle
    angles = np.linspace(start_angle, chosen_end_angle, num_points)

    # Generate points
    points = []
    for theta in angles:
        point_2d = center_2d + radius * np.array([np.cos(theta), np.sin(theta)])
        point = np.array(center)
        point[idx1] = point_2d[0]
        point[idx2] = point_2d[1]
        points.append(point)

    return np.array(points)



def reconstruct_circle_points(stroke, num_points=20):
    center = np.array(stroke[0:3])
    normal = np.array(stroke[3:6])
    radius = stroke[7]

    # Normalize the normal vector
    normal = normal / np.linalg.norm(normal)

    # Find two orthogonal vectors in the circle's plane
    if np.allclose(normal, [0, 0, 1]):
        u = np.array([1, 0, 0])
    else:
        u = np.cross(normal, [0, 0, 1])
        u /= np.linalg.norm(u)
    v = np.cross(normal, u)

    t_vals = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    pts = []

    for t in t_vals:
        point = center + radius * (np.cos(t) * u + np.sin(t) * v)
        pts.append(point)

    return np.array(pts)
