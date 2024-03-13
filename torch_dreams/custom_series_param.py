from .auto_series_param import BaseSeriesParam

import torch

from .utils import (
    lucid_colorspace_to_rgb,
    normalize,
    get_fft_series_scale,
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
    def __init__(self, series, device):
        
        super().__init__()
        self.device = device
        self.set_param(series)

    def normalize(self,x, device):
        # TODO: implement normalization
        #return normalize(x = x, device= device)
        return x

    def postprocess(self, device):
        out = fft_to_series(channels=self.channels, length=self.length, series_parameter=self.param, device=device)
        out = lucid_colorspace_to_rgb(t = out, device= device).clamp(0,1)
        return out

    def forward(self, device):
        return self.normalize(self.postprocess(device=device), device=device)

    def to_cl_tensor(self, device = 'cpu'):
        t = self.forward(device= device).squeeze(0).clamp(0,1).detach()
        return t

    def to_hwc_tensor(self, device = 'cpu'):
        t = self.forward(device= device).squeeze(0).clamp(0,1).permute(1,2,0).detach()
        return t

    def to_nchw_tensor(self, device = 'cpu'):
        """returns a tensor of shape  [1,3, height, width] which represents the image

        Args:
            device (str, optional): 'cuda' or 'cpu'. Defaults to 'cpu'.

        Returns:
            torch.tensor: NCHW tensor
        """
        t = self.forward(device= device).clamp(0,1).detach()
        return t

    def set_param(self, tensor):
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

        self.channels, self.sequence_length = tensor.shape[-2], tensor.shape[-1]
        scale = get_fft_series_scale(l=self.sequence_length, device=self.device)
        self.param = cl_series_to_fft_param(tensor.squeeze(0), device=self.device)  / scale
        self.param.requires_grad_()
        self.optimizer = None
