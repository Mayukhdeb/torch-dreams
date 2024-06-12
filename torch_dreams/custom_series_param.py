from .base_series_param import BaseSeriesParam

import numpy as np
import torch

from .utils import (
    lucid_colorspace_to_rgb,
    normalize,
    get_fft_scale_custom_series,
    cl_series_to_fft_param,
    fft_to_series
)


class CustomSeriesParam(BaseSeriesParam):
    """FFT parameterization for custom series.

    Works well with:
    * lower learning rates (3e-4) 
    * gradients clipped to (0, 0.1)
    * weight decay (1e-1)

    Args:
        series (torch.tensor): input tensor with shape [channels, length].
        device (str): 'cuda' or 'cpu'

    Example: 
    ```
    series = torch.ones((1, 2, 100))
    param = custom_series_param(series=series, device='cuda')

    result = dreamy_boi.render(
        image_parameter=param,
        layers = [model.Mixed_6c],
        lr = 3e-4,
        grad_clip = 0.1,
        weight_decay= 1e-1
    )
    ```
    """
    def __init__(
            self,
            series,
            device,
            #channel_correlation_matrix,
            normalize_mean=None,
            normalize_std=None,
    ):
        batch_size = series.shape[0]
        channels = series.shape[1]
        length = series.shape[2]

        super().__init__(
            batch_size=batch_size,
            channels=channels,
            length=length,
            param=series,  # we use set_param in the next step
            normalize_mean=normalize_mean,
            normalize_std=normalize_std,
            device=device,
        )

        channel_correlation_matrix = get_normalized_correlation_matrix(channels)

        self.set_param(series, channel_correlation_matrix, device=device)

    def postprocess(self, device):
        out = fft_to_series(
            channels=self.channels,
            length=self.length,
            series_parameter=self.param,
            device=device,
        )
        out = lucid_colorspace_to_rgb(t=out, device=device).clamp(0,1)
        return out


    def set_param(self, tensor, channel_correlation_matrix, device):
        """sets an NCL tensor as the parameter in the frequency domain,
        useful for transforming custom series between iterations.

        Use in combination with `self.to_ncl_tensor()` like:

        ```
        a = self.to_cl_tensor()
        # do something with a
        t = transforms.Compose([
            transforms.RandomScale(0,5, 1.2)
        ])
        a = t(a)
        #set as parameter again
        self.set_param(a)
        ```

        WARNING: tensor should have values clipped between 0 and 1.

        Args:
            tensor (torch.tensor): input tensor with shape [1,channels, length] and values clipped between 0,1.
        """
        assert len(tensor.shape) == 3
        assert tensor.shape[0] == 1

        self.tensor = tensor

        batch_size = tensor.shape[0]
        channels = tensor.shape[1]
        length = tensor.shape[2]

        scale = get_fft_scale_custom_series(length=length, device=device)
        # TODO: denormalize
        #fft_param = cl_series_to_fft_param(self.denormalize(tensor.squeeze(0)), device=device)
        fft_param = cl_series_to_fft_param(tensor, channel_correlation_matrix=channel_correlation_matrix, device=device)
        self.param = fft_param / scale

        self.param.requires_grad_()

        self.batch_size = batch_size
        self.channels = channels
        self.length = length
        self.device = device


def get_normalized_correlation_matrix(channels):
    # TODO: these values must be passed by the user
    correlation_svd_sqrt = np.random.rand(channels, channels).astype(np.float32)

    max_norm_svd_sqrt = np.max(np.linalg.norm(correlation_svd_sqrt, axis=0))
    correlation_normalized = torch.tensor(correlation_svd_sqrt / max_norm_svd_sqrt)
    return correlation_normalized
