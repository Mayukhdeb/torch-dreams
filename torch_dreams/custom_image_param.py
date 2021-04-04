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

def get_fft_scale_custom_img(h, w, decay_power=.75, device = 'cuda'):
    d=.5**.5 # set center frequency scale to 1
    fy = np.fft.fftfreq(h,d=d)[:,None]

    fx = np.fft.rfftfreq(w,d=d)[:(w//2)+1]   
    freqs = (fx*fx + fy*fy) ** decay_power
    scale = 1.0 / np.maximum(freqs, 1.0 / (max(w, h)*d))
    scale = torch.tensor(scale).float().to(device)

    return scale

def denormalize(x):

    return x.float()*Constants.imagenet_std[...,None,None].to(x.device) + Constants.imagenet_mean[...,None,None].to(x.device)

def rgb_to_lucid_colorspace(t, device = 'cuda'):
    t_flat = t.permute(0,2,3,1)
    inverse = torch.inverse(Constants.color_correlation_matrix.T.to(device))
    t_flat = torch.matmul(t_flat.to(device), inverse)
    t = t_flat.permute(0,3,1,2)
    return t

def chw_rgb_to_fft_param(x, device):
    im_tensor = torch.tensor(x).unsqueeze(0).float()

    x = rgb_to_lucid_colorspace(denormalize(im_tensor), device= device)

    x = torch.fft.rfft2(x, s = (x.shape[-2], x.shape[-1]), norm = 'ortho')
    return x

def fft_to_rgb_custom_img(height, width, image_parameter, device = 'cuda'):

    scale = get_fft_scale_custom_img(height, width , device= device).to(image_parameter.device)
    t = scale * image_parameter
   
    if  torch.__version__[:3] == '1.8':
        t = torch.fft.irfft2(t,  s = (height, width), norm = 'ortho')
    else:
        raise PytorchVersionError(version = torch.__version__)

    return t
    
class custom_image_param(BaseImageParam):
    """FFT parameterization for custom images 

    Works well with:
    * lower learning rates (3e-4) 
    * gradients clipped to (0, 0.1)
    * weight decay (1e-1)

    Args:
        filename (str): 'path/to/image.jpg'
        device (str): 'cuda' or 'cpu'

    Example: 
    ```
    param = custom_image_param(filename = 'image.jpg', device= 'cuda')

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
    def __init__(self, filename, device):
        
        super().__init__()
        im = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB)/255.
        self.device = device
        im_tensor_chw = torch.tensor(im).permute(-1,0,1)
        self.height, self.width = im_tensor_chw.shape[-2], im_tensor_chw.shape[-1]
        self.param = chw_rgb_to_fft_param(im_tensor_chw, device = self.device)  / get_fft_scale_custom_img(h = self.height, w = self.width, device= self.device)
        self.param.requires_grad_()
        self.optimizer = None

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