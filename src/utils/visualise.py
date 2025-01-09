# Plotting function adapted from https://github.com/google-deepmind/neural-processes

import matplotlib.pyplot as plt


def plot_functions(target_x, target_y, context_x, context_y, pred_y, var):
    """Plots the predicted mean and variance and the context points.

    Parameters
    ----------
    target_x : torch.Tensor [batch_size, num_targets, x_size]
        The x values of the target points.

    target_y : torch.Tensor [batch_size, num_targets, y_size]
        The y values of the target points.

    context_x : torch.Tensor [batch_size, num_context, x_size]
        The x values of the context points.

    context_y : torch.Tensor [batch_size, num_context, y_size]
        The y values of the context points.

    pred_y : torch.Tensor [batch_size, num_targets, y_size]
        The predicted means of the y values at the target points.

    var : torch.Tensor [batch_size, num_targets, y_size]
        The predicted variances of the y values at the target points.
    """

    # Plot everything
    plt.plot(target_x[0], pred_y[0], "b", linewidth=2)
    plt.plot(target_x[0], target_y[0], "k:", linewidth=2)
    plt.plot(context_x[0], context_y[0], "ko", markersize=10)
    plt.fill_between(
        target_x[0, :, 0],
        pred_y[0, :, 0] - var[0, :, 0],
        pred_y[0, :, 0] + var[0, :, 0],
        alpha=0.2,
        facecolor="#65c9f7",
        interpolate=True,
    )

    # Make the plot pretty
    plt.yticks([-2, 0, 2], fontsize=16)
    plt.xticks([-2, 0, 2], fontsize=16)
    plt.ylim([-2, 2])
    plt.grid(False)
    ax = plt.gca()
    ax.set_facecolor("white")
    plt.show()
