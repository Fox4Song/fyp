# Plotting function adapted from https://github.com/google-deepmind/neural-processes

import matplotlib.pyplot as plt


def plot_functions(
    target_x, target_y, context_x, context_y, pred_y, var=None, save_fig=None
):
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

    save_fig : str (optional)
        The file path to save the figure. If None, the figure is not saved.
    """

    # Plot everything
    plt.plot(target_x[0], pred_y[0], "b", linewidth=2)
    plt.plot(target_x[0], target_y[0], "k:", linewidth=2)
    plt.plot(context_x[0], context_y[0], "ko", markersize=10)
    if var is not None:
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
    if save_fig:
        plt.savefig(save_fig)
    plt.show()


def plot_model_comparisons(
    target_x,
    target_y,
    context_x,
    context_y,
    np_pred_mean,
    np_pred_std,
    tf_pred_y,
    num_context,
    save_fig=None,
):
    """
    Plots the Neural Process predictions (mean and uncertainty),
    Transformer predictions, and the context points.

    Parameters
    ----------
    target_x : np.ndarray [batch_size, num_targets, x_size]
        The x values of the target points.

    target_y : np.ndarray [batch_size, num_targets, y_size]
        The y values of the target points.

    context_x : np.ndarray [batch_size, num_context, x_size]
        The x values of the context points.

    context_y : np.ndarray [batch_size, num_context, y_size]
        The y values of the context points.

    np_pred_mean : np.ndarray [batch_size, num_targets, y_size]
        The predicted mean of the y values at the target points by the Neural Process.

    np_pred_std : np.ndarray [batch_size, num_targets, y_size]
        The predicted standard deviation of the y values at the target points by the Neural Process.

    tf_pred_y : np.ndarray [batch_size, num_targets, y_size]
        The predicted y values at the target points by the Transformer.

    num_context : int
        The number of context points.

    save_fig : str (optional)
        The file path to save the figure. If None, the figure is not saved.
    """

    # Context and target points
    plt.plot(target_x[0], target_y[0], "k:", linewidth=2)
    plt.plot(context_x[0], context_y[0], "ko", markersize=10)

    # NP predictions
    plt.plot(target_x[0], np_pred_mean[0], "b", linewidth=2, label="NP Prediction")
    plt.fill_between(
        target_x[0, :, 0],
        np_pred_mean[0, :, 0] - np_pred_std[0, :, 0],
        np_pred_mean[0, :, 0] + np_pred_std[0, :, 0],
        alpha=0.2,
        facecolor="#65c9f7",
        interpolate=True,
    )

    # Transformer predictions
    plt.plot(
        target_x[0, :, 0],
        tf_pred_y[0, :, 0],
        "r",
        linewidth=2,
        label="Transformer Prediction",
    )

    # Make the plot pretty
    plt.yticks([-2, 0, 2], fontsize=16)
    plt.xticks([-2, 0, 2], fontsize=16)
    plt.ylim([-2, 2])
    plt.title(f'Function Prediction with {num_context} Context Points', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(False)
    ax = plt.gca()
    ax.set_facecolor("white")
    if save_fig:
        plt.savefig(save_fig)
    plt.show()
