import matplotlib.pyplot as plt
import numpy as np

from .polygon_metrics import (
    side_length_mae,
    angle_mae,
    perimeter_difference,
    area_difference,
    area_relative_error,
    centroid_distance,
    iou,
    hausdorff_distance,
    chamfer_distance,
    angle_sum_error,
    length_closure_consistency,
    angle_geometry_consistency,
    prefix_mae,
)


def plot_polygon(vertices, title=None):
    """
    Plots a convex polygon using a list of vertex coordinates.

    Parameters
    ----------
    vertices : list
        A flat list of vertex coordinates in the format [x1, y1, x2, y2, ..., xn, yn].
    title : str (optional)
        The title of the plot. If not provided, defaults to "Convex Polygon".
    """
    # Extract x and y coordinates from vertices
    x_coords = [v[0] for v in vertices]
    y_coords = [v[1] for v in vertices]

    # Append the first vertex to close the polygon
    x_coords.append(vertices[0][0])
    y_coords.append(vertices[0][1])

    plt.figure(figsize=(6, 6))
    plt.plot(x_coords, y_coords, marker="o", linestyle="-")
    plt.fill(x_coords, y_coords, alpha=0.2)
    if title:
        plt.title(title)
    else:
        plt.title("Convex Polygon")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.axis("equal")
    plt.grid(True)
    plt.show()


def plot_polygon_angle_completion_task_metrics(preds, trues, title=None):
    """
    Plots the metrics for the polygon angle completion task.

    Parameters
    ----------
    preds : list [B, N]
        A list of predicted polygon angles.

    trues : list [B, N]
        A list of true polygons angles.
    """

    def angle_mae(preds, trues):
        """
        Mean Absolute Error of angles over batch.
        """
        return sum(
            abs(sum(preds[i]) - sum(trues[i])) / len(trues[i])
            for i in range(len(preds))
        ) / len(preds)

    def angle_sum_error(preds):
        """
        Sum of angles error over batch.
        """
        return sum(
            abs(sum(preds[i]) - ((len(trues[i]) - 2) * 180)) for i in range(len(preds))
        ) / len(preds)

    angle_mae_err = angle_mae(preds, trues)
    angle_sum_err = angle_sum_error(preds)

    # Plot 1: Bar chart of average angle errors
    plt.figure(figsize=(10, 6))
    plt.bar(["Angle MAE", "Angle Sum Error"], [angle_mae_err, angle_sum_err])
    plt.ylabel("MAE")
    if not title:
        title = "Angle MAE for Polygon Angle Completion Task"
    plt.title(title)
    plt.tight_layout()
    plt.show()

    # Plot 2: Compute and plot prefix token mae
    pref_acc = prefix_mae(preds, trues)
    positions = list(pref_acc.keys())
    accuracies = list(pref_acc.values())

    plt.figure(figsize=(8, 5))
    plt.plot(positions, accuracies)
    plt.xlabel("Token Position")
    plt.ylabel("Prefix Token MAE")
    plt.title("Prefix Token MAE by Position for Polygon Angle Completion Task")
    plt.grid(True)
    plt.show()


def plot_polygon_metrics(preds, trues, metrics=None):
    """
    1. Bar chart of average metric values across the batch.
    2. Line plot of prefix token accuracy by position.
    """

    metric_funcs = {
        "Side Length MAE": side_length_mae,
        "Angle MAE": angle_mae,
        "Perimeter Diff.": perimeter_difference,
        "Area Diff.": area_difference,
        "Rel. Area Err.": area_relative_error,
        "Centroid Dist.": centroid_distance,
        "IoU": iou,
        "Hausdorff Dist.": hausdorff_distance,
        "Chamfer Dist.": chamfer_distance,
        "Angle Sum Err.": angle_sum_error,
        "Length Closure Cons.": length_closure_consistency,
        "Angle Geom. Cons.": angle_geometry_consistency,
    }

    avg_metrics = {}
    for name, func in metric_funcs.items():
        vals = [func(p, t) for p, t in zip(preds, trues)]
        avg_metrics[name] = sum(vals) / len(vals)

    # Plot 1: Bar chart of average metrics
    plt.figure(figsize=(10, 6))
    plt.bar(avg_metrics.keys(), avg_metrics.values())
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Average Value")
    plt.title("Average Polygon Metric Values")
    plt.tight_layout()
    plt.show()

    # Plot 2: Compute and plot prefix token mae
    pref_acc = prefix_mae(preds, trues)
    positions = list(pref_acc.keys())
    accuracies = list(pref_acc.values())

    plt.figure(figsize=(8, 5))
    plt.plot(positions, accuracies)
    plt.xlabel("Token Position")
    plt.ylabel("Accuracy")
    plt.title("Prefix Token Accuracy by Position")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Example vertices for a polygon
    vertices = np.random.rand(5, 2).tolist()  # Random polygon with 5 vertices
    plot_polygon(vertices, title="Example Polygon")

    # Example angle predictions and ground truths
    preds = np.random.randint(0, 360, (1, 5)).tolist()
    trues = np.random.randint(0, 360, (1, 5)).tolist()
    plot_polygon_angle_completion_task_metrics(preds, trues)
