import numpy as np
import copy

def do_perturb(all_lines, stroke_node_features, 
               point_jitter_ratio=0.02, endpoint_shift_ratio=0.04, overdraw_ratio=0.1):
    """
    Perturbs only feature strokes in all_lines (1: straight, 3: arc).
    
    Parameters:
    - all_lines: list of dicts with 'geometry' key (list of 3D points)
    - stroke_node_features: (num_strokes, 11) array-like
    - *_ratio: proportional values w.r.t. stroke length

    Returns:
    - new_all_lines: copy of all_lines with perturbed geometries
    """
    new_all_lines = copy.deepcopy(all_lines)

    for i, stroke in enumerate(new_all_lines):

        # straight stroke perturbation
        if stroke_node_features[i][-1] == 1:
            geometry = stroke["geometry"]
            if len(geometry) < 2 or len(geometry) > 5:
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

            stroke["geometry"] = pts.tolist()

        # arc perturbation
        if stroke_node_features[i][-1] == 3:
            geometry = stroke["geometry"]
            if len(geometry) < 3:
                continue

            perturbed = perturb_arc_by_interpolation(
                geometry,
                arc_fraction=np.random.uniform(0.3, 0.6),  # people usually stop short
                noise_scale_ratio=0.0005
            )
            stroke["geometry"] = perturbed.tolist()

        # circle perturbation
        if stroke_node_features[i][-1] == 2:
            geometry = stroke["geometry"]
            if len(geometry) < 6:
                continue

            perturbed = perturb_circle_geometry(np.array(geometry))
            stroke["geometry"] = perturbed.tolist()


    return new_all_lines


def perturb_arc_by_interpolation(pts, t_range=np.pi/2, num_points=None,
                                  arc_fraction=None, noise_scale_ratio=0.0001,
                                  endpoint_shift_ratio=0.002):
    """
    Simulate human-drawn arc by interpolating between a straight line and a circular arc.

    Parameters:
    - pts: original arc polyline
    - arc_fraction: strength of curvature [0=line, 1=full arc], sampled randomly if None
    - noise_scale_ratio: wobble applied to inner points (fraction of radius)
    - endpoint_shift_ratio: small random movement for start/end

    Returns:
    - np.ndarray of perturbed arc points
    """
    pts = np.array(pts)
    if num_points is None:
        num_points = len(pts)

    start = pts[0]
    end = pts[-1]
    mid = pts[len(pts) // 2]

    # Estimate arc plane and radius
    v1 = mid - start
    v2 = mid - end
    normal = np.cross(v1, v2)
    normal /= np.linalg.norm(normal) + 1e-8

    chord = end - start
    chord_len = np.linalg.norm(chord)
    R = chord_len / np.sqrt(2)

    chord_mid = (start + end) / 2
    perp = np.cross(normal, chord)
    perp /= np.linalg.norm(perp) + 1e-8
    center = chord_mid + R / 2 * perp

    # Local basis
    u = (start - center)
    u /= np.linalg.norm(u) + 1e-8
    v = np.cross(normal, u)

    # Apply small random shift to start and end
    start += np.random.normal(scale=R * endpoint_shift_ratio, size=3)
    end += np.random.normal(scale=R * endpoint_shift_ratio, size=3)

    # Arc strength
    if arc_fraction is None:
        arc_fraction = np.random.uniform(0.3, 0.9)

    noise_scale = R * noise_scale_ratio
    t_vals = np.linspace(0, t_range, num_points)

    arc_points = []
    for j, t in enumerate(t_vals):
        arc_pt = center + R * (np.cos(t) * u + np.sin(t) * v)
        line_pt = start + (end - start) * (j / (num_points - 1))
        pt = (1 - arc_fraction) * line_pt + arc_fraction * arc_pt

        if 0 < j < len(t_vals) - 1:
            pt += np.random.normal(scale=noise_scale, size=3)

        arc_points.append(pt)

    return np.array(arc_points)




def perturb_circle_geometry(pts):
    """
    Perturb a clean circle to resemble a human-drawn ellipse, with smooth imperfections.
    """
    pts = np.array(pts)
    N = len(pts)
    if N < 6:
        return pts

    # Estimate best-fit plane
    center = pts.mean(axis=0)
    centered = pts - center
    _, _, vh = np.linalg.svd(centered)
    u, v, normal = vh[0], vh[1], vh[2]

    # Project to 2D and get radius
    coords_2d = np.array([[np.dot(p - center, u), np.dot(p - center, v)] for p in pts])
    radius = np.mean(np.linalg.norm(coords_2d, axis=1))

    # === Randomized parameters ===
    rx = radius * np.random.uniform(0.8, 1.2)
    ry = radius * np.random.uniform(0.8, 1.2)
    theta = np.random.uniform(0, 2 * np.pi)
    noise_scale = np.random.uniform(0.001, 0.005) * radius
    shift_last_point = np.random.uniform(0.4, 0.8) * radius

    # Rotation matrix
    rot = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)]
    ])

    # Generate ellipse
    t_vals = np.linspace(0, 2 * np.pi, N, endpoint=False)
    ellipse = np.stack([rx * np.cos(t_vals), ry * np.sin(t_vals)], axis=1)
    ellipse = ellipse @ rot.T
    ellipse += np.random.normal(scale=noise_scale, size=ellipse.shape)

    # Back to 3D
    new_pts = np.array([center + x * u + y * v for x, y in ellipse])

    # Spread final distortion across last ~5 points smoothly
    distortion = np.random.normal(scale=shift_last_point, size=3)
    decay_weights = np.linspace(1.0, 0.9, 5)
    for k, w in enumerate(decay_weights):
        idx = -2 - k
        if idx >= 0:
            new_pts[idx] += w * distortion


    # === Add extension line beyond the circle ===
    num_extra_points = np.random.randint(1, 4)  # 1 to 3 extra points
    extension_spacing = np.random.uniform(0.05, 0.1) * radius

    # Tangent direction at end
    tangent = new_pts[-1] - new_pts[-2]
    tangent /= np.linalg.norm(tangent) + 1e-8

    for i in range(1, num_extra_points + 1):
        extension_point = new_pts[-1] + i * extension_spacing * tangent
        new_pts = np.vstack([new_pts, extension_point])

    return new_pts
