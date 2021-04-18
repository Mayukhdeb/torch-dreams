from .auto_image_param import BaseImageParam

import cv2 
import torch
import numpy as np

from .constants import Constants
from .error_handlers import PytorchVersionError

from .utils import (
    lucid_colorspace_to_rgb, 
    normalize,
    get_fft_scale_custom_img,
    denormalize,
    rgb_to_lucid_colorspace,
    chw_rgb_to_fft_param,
    fft_to_rgb_custom_img
)

class custom_image_param(BaseImageParam):
    """FFT parameterization for custom images 

    Works well with:
    * lower learning rates (3e-4) 
    * gradients clipped to (0, 0.1)
    * weight decay (1e-1)

    Args:
        image (str or torch.tensor): 'path/to/image.jpg' or input tensor with shape [1,3, height, width] and values clipped between 0,1.
        device (str): 'cuda' or 'cpu'

    Example: 
    ```
    param = custom_image_param(image = 'image.jpg', device= 'cuda')

    image_param = dreamy_boi.render(
        image_parameter= param,
        layers = [model.Mixed_6c],
        lr = 3e-4,
        grad_clip = 0.1,
        weight_decay= 1e-1
    )

    image_param.save('saved.jpg')
    ```
    """
    def __init__(self, image, device):
        
        super().__init__()
        self.device = device
        if isinstance(image, str):
            image = cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB)/255.
            image = torch.tensor(image).permute(-1,0,1).unsqueeze(0)
        self.set_param(image)

    def normalize(self,x, device):
        return normalize(x = x, device= device)

    def postprocess(self, device):
        out = fft_to_rgb_custom_img(height = self.height, width = self.width, image_parameter= self.param, device= device)
        out = lucid_colorspace_to_rgb(t = out, device= device).clamp(0,1)
        return out

    def forward(self, device):
        return self.normalize(self.postprocess(device = device), device= device).clamp(0,1)

    def to_chw_tensor(self, device = 'cpu'):
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
        """sets an NCHW tensor as the parameter in the frequency domain, 
        useful for transforming custom images between iterations.

        Use in combination with `self.to_nchw_tensor()` like:

        ```
        a = self.to_nchw_tensor()
        # do something with a
        t = transforms.Compose([
            transforms.RandomRotation(45)
        ])
        a = t(a)
        #set as parameter again
        self.set_param(a)
        ```

        WARNING: tensor should have values clipped between 0 and 1. 

        Args:
            tensor (torch.tensor): input tensor with shape [1,3, height, width] and values clipped between 0,1.
        """
        assert len(tensor.shape) == 4

        self.height, self.width = tensor.shape[-2], tensor.shape[-1]
        self.param = chw_rgb_to_fft_param(tensor.squeeze(0), device = self.device)  / get_fft_scale_custom_img(h = self.height, w = self.width, device= self.device)
        self.param.requires_grad_()
        self.optimizer = None
