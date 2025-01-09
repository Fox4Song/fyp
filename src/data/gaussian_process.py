# GP dataset adapted from https://github.com/google-deepmind/neural-processes

import torch
import torch.nn as nn


class GPCurvesReader(nn.Module):
    """Creates a regression dataset of functions sampled from a Gaussian Process (GP)

    Supports vector inputs (x) and vector outputs (y). Kernel is
    mean-squared exponential, using the x-value l2 coordinate distance scaled by
    some factor chosen randomly in a range. Outputs are independent gaussian
    processes.

    Parameters
    ----------
    batch_size : int
        Number of functions to sample

    max_num_context : int
        Maximum number of observations in the context

    x_size : int
        Number of x values

    y_size : int
        Number of y values

    l1_scale : float
        Kernel distance function scale

    sigma_scale : float
        Kernel variance scale

    testing : bool
        If True, always use the same context points
    """

    def __init__(
        self,
        batch_size,
        max_num_context,
        x_size=1,
        y_size=1,
        l1_scale=0.4,
        sigma_scale=1.0,
        testing=False,
    ):
        super(GPCurvesReader, self).__init__()

        self._batch_size = batch_size
        self._max_num_context = max_num_context
        self._x_size = x_size
        self._y_size = y_size
        self._l1_scale = l1_scale
        self._sigma_scale = sigma_scale
        self._testing = testing

    def _gaussian_kernel(self, xdata, l1, sigma_f, sigma_noise=2e-2):
        """Applies the Gaussian kernel to generate curve data.

        Parameters
        ----------
        xdata : torch.Tensor [B, num_total_points, x_size]
            x data points

        l1 : torch.Tensor [B, y_size, x_size]
            l1 scales

        sigma_f : torch.Tensor [B, y_size]
            Magnitude of the std

        sigma_noise : float
            Std noise parameter (for stability)

        Returns
        -------
        torch.Tensor [B, y_size, num_total_points, num_total_points]
            The kernel, a positive-definite covariance matrix
        """

        num_total_points = xdata.shape[1]

        # Expand and take the difference
        # [batch_size, 1, num_total_points, x_size]
        xdata1 = xdata.unsqueeze(1)
        # [batch_size, num_total_points, 1, x_size]
        xdata2 = xdata.unsqueeze(2)
        # [batch_size, num_total_points, num_total_points, x_size]
        diff = xdata1 - xdata2

        # [batch_size, y_size, num_total_points, num_total_points, x_size]
        norm = (diff[:, None, :, :, :] / l1[:, :, None, None, :]) ** 2

        # [batch_size, y_size, num_total_points, num_total_points]
        norm = torch.sum(norm, dim=-1)

        kernel = (sigma_f**2)[:, :, None, None] * torch.exp(-0.5 * norm)

        # Add some noise to the diagonal to make the cholesky work.
        kernel += (sigma_noise**2) * torch.eye(num_total_points)

        assert kernel.shape == (
            self._batch_size,
            self._y_size,
            num_total_points,
            num_total_points,
        )
        return kernel

    def generate_curves(self):
        """Builds the op delivering the data.

        Generated functions are float32 tensors with x values between -2 and 2.

        Returns
        -------
        context_x : torch.Tensor [B, num_context, x_size]
            The x values of the context points

        context_y : torch.Tensor [B, num_context, y_size]
            The y values of the context points

        target_x : torch.Tensor [B, num_target, x_size]
            The x values of the target points

        target_y : torch.Tensor [B, num_target, y_size]
            The y values of the target points

        num_total_points : int
            The total number of points

        num_context_points : int
            The number of context points
        """

        num_context = torch.randint(low=3, high=self._max_num_context + 1, size=(1,))

        # If we are testing we want to have more targets and have them evenly
        # distributed in order to plot the function.
        if self._testing:
            num_target = 400
            num_total_points = num_target
            # [1, num_total_points]
            x_values = torch.arange(-2.0, 2.0, 1.0 / 100).unsqueeze(0)
            # [batch_size, num_total_points, 1]
            x_values = x_values.repeat(self._batch_size, 1).unsqueeze(-1)
        # If we are training we sample the number of target points.
        else:
            num_target = torch.randint(low=2, high=self._max_num_context + 1, size=(1,))
            num_total_points = num_context + num_target
            x_values = torch.FloatTensor(
                self._batch_size, num_total_points, self._x_size
            ).uniform_(-2, 2)

        # Set kernel parameters
        l1 = torch.ones(self._batch_size, self._y_size, self._x_size) * self._l1_scale
        sigma_f = torch.ones(self._batch_size, self._y_size) * self._sigma_scale

        kernel = self._gaussian_kernel(x_values, l1, sigma_f)

        # Calculate Cholesky using double precision for better stability
        cholesky = torch.linalg.cholesky(kernel.double()).float()

        # Sample a curve
        # [batch_size, y_size, num_total_points, 1]
        y_values = torch.matmul(
            cholesky, torch.randn(self._batch_size, self._y_size, num_total_points, 1)
        )

        # [batch_size, num_total_points, y_size]
        y_values = torch.transpose(y_values.squeeze(-1), -1, -2)

        if self._testing:
            target_x = x_values
            target_y = y_values

            idx = torch.randperm(num_target)
            context_x = x_values[:, idx[:num_context], :]
            context_y = y_values[:, idx[:num_context], :]

        else:
            # Select the targets which will consist of the context points as well as
            # some new target points
            target_x = x_values[:, : num_target + num_context, :]
            target_y = y_values[:, : num_target + num_context, :]

            context_x = x_values[:, :num_context, :]
            context_y = y_values[:, :num_context, :]

        return (
            context_x,
            context_y,
            target_x,
            target_y,
            target_x.shape[1],
            num_context,
        )
