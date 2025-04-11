import numpy as np
import json
from scipy.spatial import cKDTree
from pathlib import Path

from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopoDS import topods
from OCC.Core.TopAbs import TopAbs_FACE
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.gp import gp_Trsf
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform


def read_step(filepath):
    try:
        step_reader = STEPControl_Reader()
        step_reader.ReadFile(str(filepath))
        step_reader.TransferRoot(1)
        shape = step_reader.Shape()
        return shape
    except Exception as e:
        print(f"Error reading STEP file: {filepath}, {e}")
        return None


def sample_points_from_shape(shape, tolerance=0.0001, sample_density=100):
    if shape is None:
        print("Shape is None, skipping sampling.")
        return np.array([])
    try:
        BRepMesh_IncrementalMesh(shape, tolerance)
        points = []
        explorer = TopExp_Explorer(shape, TopAbs_FACE)
        while explorer.More():
            face = topods.Face(explorer.Current())
            adaptor = BRepAdaptor_Surface(face)
            umin, umax = adaptor.FirstUParameter(), adaptor.LastUParameter()
            vmin, vmax = adaptor.FirstVParameter(), adaptor.LastVParameter()
            u_step = (umax - umin) / sample_density
            v_step = (vmax - vmin) / sample_density
            u = umin
            while u <= umax:
                v = vmin
                while v <= vmax:
                    point = adaptor.Value(u, v)
                    points.append((point.X(), point.Y(), point.Z()))
                    v += v_step
                u += u_step
            explorer.Next()
        return np.array(points)
    except Exception as e:
        print(f"Error sampling points from shape: {e}")
        return np.array([])


def chamfer_distance(points1, points2):
    if points1.shape[0] == 0 or points2.shape[0] == 0:
        print("Empty point cloud detected!")
        return float('inf')
    try:
        tree = cKDTree(points2)
        dist, _ = tree.query(points1)
        return np.mean(dist)
    except Exception as e:
        print(f"Error computing Chamfer distance: {e}")
        return float('inf')


def apply_transform_from_json(shape, matrix_path):
    """Apply 4x4 transform matrix (from JSON file) to the shape."""
    try:
        with open(matrix_path, 'r') as f:
            matrix = json.load(f)

        r00, r01, r02, tx = matrix[0]
        r10, r11, r12, ty = matrix[1]
        r20, r21, r22, tz = matrix[2]

        trsf = gp_Trsf()
        trsf.SetValues(
            r00, r01, r02, tx,
            r10, r11, r12, ty,
            r20, r21, r22, tz
        )
        transform = BRepBuilderAPI_Transform(shape, trsf, True)
        return transform.Shape()
    except Exception as e:
        print(f"Error applying transform: {e}")
        return shape


def compute_fidelity_score(gt_brep_path, output_brep_path, tolerance=0.0001, sample_density=10):
    """
    Computes the fidelity score based on Chamfer distances between two BREP files.
    """
    try:
        # Read shapes
        gt_shape = read_step(gt_brep_path)
        output_shape = read_step(output_brep_path)

        # Apply transformation to GT shape
        matrix_path = Path(gt_brep_path).parent / "gt_canvas" / "matrix.json"
        gt_shape = apply_transform_from_json(gt_shape, matrix_path)

        if gt_shape is None or output_shape is None:
            print("Invalid shape detected, skipping fidelity computation.")
            return 0

        # Sample points
        gt_points = sample_points_from_shape(gt_shape, tolerance, sample_density)
        output_points = sample_points_from_shape(output_shape, tolerance, sample_density)

        if gt_points.shape[0] == 0 or output_points.shape[0] == 0:
            print("Insufficient points sampled, skipping fidelity computation.")
            return 0

        # Compute directional Chamfer distances
        gt_to_output = chamfer_distance(gt_points, output_points) * 1000
        output_to_gt = chamfer_distance(output_points, gt_points) * 1000

        fidelity_score = 1 / (1 + gt_to_output + output_to_gt)
        return fidelity_score
    except Exception as e:
        print(f"Error computing fidelity score: {e}")
        return 0
