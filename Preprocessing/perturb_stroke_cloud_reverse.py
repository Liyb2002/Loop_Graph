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

            stroke_cloud.append(pts)

        elif stroke_type == 3:  # Arc
            pts = reconstruct_arc_points(stroke, num_points=num_points_arc)
            stroke_cloud.append(pts)

        elif stroke_type == 2:  # Circle
            pts = reconstruct_circle_points(stroke, num_points=num_points_circle)
            stroke_cloud.append(pts)

    return stroke_cloud



def reconstruct_arc_points(stroke, num_points=10):
    start = np.array(stroke[0:3])
    end = np.array(stroke[3:6])
    center = np.array(stroke[7:10])  # <-- correct indexing

    # Vectors from center to start and center to end
    vec_start = start - center
    vec_end = end - center

    # Normalize them
    vec_start /= np.linalg.norm(vec_start) + 1e-8
    vec_end /= np.linalg.norm(vec_end) + 1e-8

    radius = np.linalg.norm(start - center)

    # Create interpolation between vec_start and vec_end
    t_vals = np.linspace(0, 1, num_points)

    pts = []
    for t in t_vals:
        vec = (1 - t) * vec_start + t * vec_end
        vec /= np.linalg.norm(vec) + 1e-8
        pt = center + radius * vec
        pts.append(pt)

    return np.array(pts)




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
