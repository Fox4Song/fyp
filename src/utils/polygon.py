import matplotlib.pyplot as plt


def plot_polygon(vertices, title=None):
    """
    Plots a convex polygon using a flat list of vertex coordinates.

    Parameters
    ----------
    vertices : list
        A flat list of vertex coordinates in the format [x1, y1, x2, y2, ..., xn, yn].
    title : str (optional)
        The title of the plot. If not provided, defaults to "Convex Polygon".
    """
    # Extract x and y coordinates from vertices
    x_coords = vertices[0::2]
    y_coords = vertices[1::2]

    # Append the first vertex to close the polygon
    x_coords.append(vertices[0])
    y_coords.append(vertices[1])

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
