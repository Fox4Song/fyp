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


def segmentwise_token_mae(pred_tokens, true_tokens):
    """
    Mean absolute error over each segment: vertices, lengths, angles.
    """
    n = (len(true_tokens) - 5) // 4
    idx_sv = 1
    idx_sl = 2 + 2 * n
    idx_sa = 3 + 3 * n
    idx_e = -1

    # extract segments
    verts_true = true_tokens[idx_sv + 1 : idx_sl]
    lens_true = true_tokens[idx_sl + 1 : idx_sa]
    angs_true = true_tokens[idx_sa + 1 : idx_e]

    verts_pred = pred_tokens[idx_sv + 1 : idx_sl]
    lens_pred = pred_tokens[idx_sl + 1 : idx_sa]
    angs_pred = pred_tokens[idx_sa + 1 : idx_e]

    def mae(a, b):
        return (
            float(sum(abs(x - y) for x, y in zip(a, b)) / len(b)) if b else float("nan")
        )

    return {
        "vertex_mae": mae(verts_pred, verts_true),
        "length_mae": mae(lens_pred, lens_true),
        "angle_mae": mae(angs_pred, angs_true),
    }


def angle_sum_error(poly):
    """
    Absolute error of sum of interior angles vs (n-2)*180.
    """
    total = sum(poly.angles)
    expected = (poly.n - 2) * 180.0
    return abs(total - expected)


def length_closure_consistency(poly):
    """
    Mean absolute error between token lengths and geometry-computed side lengths.
    """
    verts = poly.vertices
    n = poly.n
    comp = [
        math.hypot(
            verts[(i + 1) % n][0] - verts[i][0], verts[(i + 1) % n][1] - verts[i][1]
        )
        for i in range(n)
    ]
    errors = [abs(a - b) for a, b in zip(poly.lengths, comp)]
    return sum(errors) / len(errors)


def angle_geometry_consistency(poly):
    """
    Mean absolute error between token angles and geometry-computed interior angles.
    """
    verts = poly.vertices
    n = poly.n
    comp_angles = []
    for i in range(n):
        prev = verts[i - 1]
        curr = verts[i]
        nxt = verts[(i + 1) % n]
        v1 = (prev[0] - curr[0], prev[1] - curr[1])
        v2 = (nxt[0] - curr[0], nxt[1] - curr[1])
        dot = v1[0] * v2[0] + v1[1] * v2[1]
        m1 = math.hypot(*v1)
        m2 = math.hypot(*v2)
        if m1 * m2 < 1e-8:
            comp_angles.append(0.0)
        else:
            cosv = max(-1.0, min(1.0, dot / (m1 * m2)))
            comp_angles.append(math.degrees(math.acos(cosv)))
    errors = [abs(a - b) for a, b in zip(poly.angles, comp_angles)]
    return sum(errors) / len(errors)


def prefix_mae(preds, trues):
    """
    Computes token MAE at each position across a batch of predicted vs true token lists.
    Returns a dict mapping position index -> MAE.
    """
    max_len = max(len(t) for t in trues)
    results = {}
    for i in range(max_len):
        errors = []
        for p, t in zip(preds, trues):
            # only consider samples where both true and pred have a token at i
            if len(t) > i and len(p) > i:
                errors.append(abs(p[i] - t[i]))
        results[i] = sum(errors) / len(errors)
    return results
