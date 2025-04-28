import random
import numpy as np

def stroke_node_features_to_polyline(stroke_node_features):
    """
    Convert stroke_node_features into a list of stroke objects (polyline format).

    Each stroke object has:
    - type: "feature_line"
    - feature_id: 0 (for now)
    - geometry: list of 3D points
    - id: unique id per stroke
    - opacity: random between 0.5 and 0.8
    """
    stroke_cloud = build_stroke_cloud_from_node_features(stroke_node_features)

    stroke_objects = []

    for idx, pts in enumerate(stroke_cloud):
        stroke_obj = {
            "type": "feature_line",
            "feature_id": 0,
            "geometry": pts.tolist(),
            "id": idx,
            "opacity": random.uniform(0.5, 0.8)
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

            jitter = np.random.normal(scale=0.01 * np.linalg.norm(end - start), size=pts.shape)
            pts[1:-1] += jitter[1:-1]  # Only jitter interior points
            stroke_cloud.append(pts)

        elif stroke_type == 3:  # Arc
            pts = reconstruct_arc_points(stroke, num_points=num_points_arc)
            # pts = perturb_arc_by_interpolation(pts)
            stroke_cloud.append(pts)

        elif stroke_type == 2:  # Circle
            pts = reconstruct_circle_points(stroke, num_points=num_points_circle)
            # pts = perturb_circle_geometry(pts)
            stroke_cloud.append(pts)

    return stroke_cloud


def reconstruct_arc_points(stroke, num_points=10):
    start = np.array(stroke[0:3])
    end = np.array(stroke[3:6])

    chord = end - start
    chord_mid = (start + end) / 2
    radius = np.linalg.norm(chord) / np.sqrt(2)

    normal = np.cross(start - chord_mid, end - chord_mid)
    normal /= np.linalg.norm(normal) + 1e-8

    perp = np.cross(normal, chord)
    perp /= np.linalg.norm(perp) + 1e-8
    center = chord_mid + radius / 2 * perp

    u = start - center
    u /= np.linalg.norm(u) + 1e-8
    v = np.cross(normal, u)

    t_vals = np.linspace(0, np.pi/2, num_points)

    pts = []
    for t in t_vals:
        pt = center + radius * (np.cos(t) * u + np.sin(t) * v)
        pts.append(pt)

    return np.array(pts)

def reconstruct_circle_points(stroke, num_points=20):
    center = np.array(stroke[0:3])
    radius = stroke[7]

    t_vals = np.linspace(0, 2 * np.pi, num_points, endpoint=False)

    pts = []
    for t in t_vals:
        x = center[0] + radius * np.cos(t)
        y = center[1] + radius * np.sin(t)
        z = center[2]
        pts.append([x, y, z])

    return np.array(pts)
