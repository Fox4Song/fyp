import math
import numpy as np
from shapely.geometry import Polygon as ShapelyPolygon


def side_length_mae(pred, true):
    """
    Mean absolute error between corresponding side lengths.
    """
    Lp = pred.lengths
    Lt = true.lengths
    if len(Lp) != len(Lt):
        raise ValueError("Predicted and true polygons have different vertex counts")
    return float(sum(abs(a - b) for a, b in zip(Lp, Lt)) / len(Lt))


def angle_mae(pred, true):
    """
    Mean absolute error between corresponding interior angles (in degrees).
    """
    Ap = pred.angles
    At = true.angles
    if len(Ap) != len(At):
        raise ValueError("Predicted and true polygons have different vertex counts")
    return float(sum(abs(a - b) for a, b in zip(Ap, At)) / len(At))


def perimeter_difference(pred, true):
    """
    Absolute difference in total perimeter.
    """
    return abs(sum(pred.lengths) - sum(true.lengths))


def area_difference(pred, true):
    """
    Absolute difference in polygon area.
    """
    sp = ShapelyPolygon(pred.vertices)
    st = ShapelyPolygon(true.vertices)
    return abs(sp.area - st.area)


def area_relative_error(pred, true):
    """
    Relative area error: |A_pred - A_true| / A_true
    """
    true_area = ShapelyPolygon(true.vertices).area
    if true_area == 0:
        return float("nan")
    return area_difference(pred, true) / true_area


def centroid_distance(pred, true):
    """
    Euclidean distance between centroids.
    """
    sp = ShapelyPolygon(pred.vertices)
    st = ShapelyPolygon(true.vertices)
    cx_p, cy_p = sp.centroid.x, sp.centroid.y
    cx_t, cy_t = st.centroid.x, st.centroid.y
    return math.hypot(cx_p - cx_t, cy_p - cy_t)


def iou(pred, true):
    """
    Intersection-over-Union of the two shapes.
    """
    sp = ShapelyPolygon(pred.vertices)
    st = ShapelyPolygon(true.vertices)
    inter = sp.intersection(st).area
    uni = sp.union(st).area
    if uni == 0:
        return float("nan")
    return inter / uni


def hausdorff_distance(pred, true):
    """
    Symmetric Hausdorff distance between the two polygons.
    """
    sp = ShapelyPolygon(pred.vertices)
    st = ShapelyPolygon(true.vertices)
    return sp.hausdorff_distance(st)


def chamfer_distance(pred, true):
    """
    Bidirectional average distance between vertex sets.
    """
    A = pred.vertices
    B = true.vertices

    def avg_min(A_list, B_list):
        return sum(
            min(math.hypot(x - x2, y - y2) for (x2, y2) in B_list) for (x, y) in A_list
        ) / len(A_list)

    return float((avg_min(A, B) + avg_min(B, A)) / 2.0)
