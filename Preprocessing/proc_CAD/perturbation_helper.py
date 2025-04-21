import numpy as np
import copy

def do_perturb(all_lines, stroke_node_features, 
               point_jitter_ratio=0.02, endpoint_shift_ratio=0.04, overdraw_ratio=0.1):
    """
    Perturbs only feature strokes (where stroke_node_features[i][-1] == 1) in a new all_lines.

    Parameters:
    - all_lines: list of dicts with 'geometry' key (list of 3D points)
    - stroke_node_features: (num_strokes, 11) array-like, last column indicates feature stroke
    - *_ratio: proportional values w.r.t. stroke length

    Returns:
    - new_all_lines: copy of all_lines with geometry of selected strokes perturbed
    """
    new_all_lines = copy.deepcopy(all_lines)

    for i, stroke in enumerate(new_all_lines):
        # Only perturb if this is a feature stroke
        if stroke_node_features[i][-1] != 1:
            continue

        geometry = stroke["geometry"]
        if len(geometry) < 2:
            continue

        pts = np.array(geometry)
        stroke_length = np.linalg.norm(pts[0] - pts[-1])
        if stroke_length < 1e-8:
            continue

        # Proportional perturbation magnitudes
        point_jitter = point_jitter_ratio * stroke_length
        endpoint_shift = endpoint_shift_ratio * stroke_length
        overdraw = overdraw_ratio * stroke_length

        # Perturb all points
        for j in range(len(pts)):
            if j == 0 or j == len(pts) - 1:
                shift = np.random.uniform(-endpoint_shift, endpoint_shift, size=3)
            else:
                shift = np.random.normal(scale=point_jitter, size=3)
            pts[j] += shift

        # Overdraw
        vec_start = pts[1] - pts[0]
        vec_end = pts[-2] - pts[-1]
        vec_start /= np.linalg.norm(vec_start) + 1e-8
        vec_end /= np.linalg.norm(vec_end) + 1e-8
        pts[0] -= overdraw * vec_start
        pts[-1] -= overdraw * vec_end

        # Save updated geometry
        stroke["geometry"] = pts.tolist()

    return new_all_lines
