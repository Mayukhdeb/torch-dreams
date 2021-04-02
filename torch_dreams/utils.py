import numpy as np
import torch
from torch import tensor
from torchvision import transforms

from .image_transforms import resize_4d_tensor_by_size
from .error_handlers import PytorchVersionError
from .constants import Constants


def init_image_param(height , width, sd=0.01, device = 'cuda'):
    """Initializes an image parameter in the frequency domain

    Args:
        height (int): height of image
        width (int): width of image
        sd (float, optional): Standard deviation of pixel values. Defaults to 0.01.
        device (str): 'cpu' or 'cuda'

    Returns:
        torch.tensor: image param to backpropagate on
    """
    img_buf = np.random.normal(size=(1, 3, height, width), scale=sd).astype(np.float32)
    spectrum_t = tensor(img_buf).float().to(device)
    return spectrum_t

def get_fft_scale(h, w, decay_power=.75, device = 'cuda'):
    d=.5**.5 # set center frequency scale to 1
    fy = np.fft.fftfreq(h,d=d)[:,None]

    if w %2 ==1:
        fx = np.fft.rfftfreq(w,d=d)[: (w+1) // 2]   
    else:

        fx = np.fft.rfftfreq(w,d=d)[: w // 2]     

    freqs = (fx*fx + fy*fy) ** decay_power
    scale = 1.0 / np.maximum(freqs, 1.0 / (max(w, h)*d))
    scale = tensor(scale).float().to(device)

    return scale


def fft_to_rgb(height, width, image_parameter, device = 'cuda'):
    """convert image param to NCHW 

    WARNING: torch v1.7.0 works differently from torch v1.8.0 on fft. 
    Hence you might find some weird workarounds in this function.

    Latest docs: https://pytorch.org/docs/stable/fft.html

    Also refer:
        https://github.com/pytorch/pytorch/issues/49637

    Args:
        height (int): height of image
        width (int): width of image 
        image_parameter (auto_image_param): instance of class auto_image_param()

    Returns:
        torch.tensor: NCHW tensor

    """
    scale = get_fft_scale(height, width, device= device).to(image_parameter.device)
    # print(scale.shape, image_parameter.shape)
    if width %2 ==1:
        image_parameter = image_parameter.reshape(1,3,height, (width+1)//2, 2)
    else:
        image_parameter = image_parameter.reshape(1,3,height, width//2, 2)

    image_parameter = torch.complex(image_parameter[..., 0], image_parameter[..., 1])
    t = scale * image_parameter

   
    if  torch.__version__[:3] == '1.8':

        t = torch.fft.irfft2(t,  s = (height, width), norm = 'ortho')
    else:
        raise PytorchVersionError(version = torch.__version__)

    return t


def lucid_colorspace_to_rgb(t,device = 'cuda'):

    t_flat = t.permute(0,2,3,1)
    t_flat = torch.matmul(t_flat.to(device) , Constants.color_correlation_matrix.T.to(device))
    t = t_flat.permute(0,3,1,2)
    return t

def rgb_to_lucid_colorspace(t, device = 'cuda'):
    t_flat = t.permute(0,2,3,1)
    inverse = torch.inverse(color_correlation_normalized().T.to(device))
    t_flat = torch.matmul(t_flat.to(device), inverse)
    t = t_flat.permute(0,3,1,2)
    return t


def denormalize(x):

    return x.float()*Constants.imagenet_std[...,None,None].to(x.device) + Constants.imagenet_mean[...,None,None].to(x.device)

def normalize(x, device = 'cuda'):
    return (x-Constants.imagenet_mean[...,None,None].to(device)) / Constants.imagenet_std[...,None,None].to(device)

def image_buf_to_rgb(h, w, img_buf, device = 'cuda', sigmoid = True):
    """[summary]

    Args:
        h (int): height 
        w (int): width
        img_buf (torch.tensor): Image parameter in frequency domain
        device (str, optional): Defaults to 'cuda'.
        sigmoid (bool, optional): Set to False when using custom images. Defaults to True.

    Returns:
        torch.tensor of shape: [C, H, W]
    """
    img = img_buf.detach()
    img = fft_to_rgb(h, w, img, device = device)
    img = lucid_colorspace_to_rgb(img, device=  device)

    if sigmoid is True:
        img = torch.sigmoid(img)
    
    img = resize_4d_tensor_by_size(img, height = h, width = w)
    
    img = img[0]    
    return img
    