import numpy as np
import torch
from torch import tensor
from torchvision import transforms

def init_image_param(height , width, sd=0.01, device = 'cuda'):
    """Initializes an image parameter 

    Args:
        height (int): height of image
        width (int): width of image
        sd (float, optional): Standard deviation of pixel values. Defaults to 0.01.
        device (str): 'cpu' or 'cuda'

    Returns:
        torch.tensor: image param to backpropagate on
    """
    img_buf = np.random.normal(size=(1, 3, height, width//2 + 1, 2), scale=sd).astype(np.float32)
    spectrum_t = tensor(img_buf).float().to(device)
    return spectrum_t

def get_fft_scale(h, w, decay_power=.75, device = 'cuda'):
    d=.5**.5 # set center frequency scale to 1
    fy = np.fft.fftfreq(h,d=d)[:,None]
    if w % 2 == 1:
        fx = np.fft.fftfreq(w,d=d)[: w // 2 + 2]
    else:
        fx = np.fft.fftfreq(w,d=d)[: w // 2 + 1]        
    freqs = (fx*fx + fy*fy) ** decay_power
    scale = 1.0 / np.maximum(freqs, 1.0 / (max(w, h)*d))
    scale = tensor(scale).float()[None,None,...,None].to(device)
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

    size log:
        before: 
            torch.Size([1, 3, height, width//2, 2]) 
            OR 
            torch.Size([1, 3, height, width+1//2, 2])

        after: 
            torch.Size([1, 3, height, width])

    """
    scale = get_fft_scale(height, width, device = device)

    t = scale * image_parameter.to(device)

    if torch.__version__[:3] == "1.7":
        t = torch.irfft(t, 2, normalized=True, signal_sizes=(height,width))
    elif  torch.__version__[:3] == '1.8':
        """
        hacky workaround to fix issues for the new torch.fft on torch 1.8.0

        """
        t = torch.complex(t[..., 0], t[..., 1])
        t = torch.fft.irfftn(t, s = (3, height, width), dim = (1,2,3), norm = 'ortho')

    return t

def color_correlation_normalized():
    color_correlation_svd_sqrt = np.asarray([[0.26, 0.09, 0.02],
                                             [0.27, 0.00, -0.05],
                                             [0.27, -0.09, 0.03]]).astype(np.float32)
    max_norm_svd_sqrt = np.max(np.linalg.norm(color_correlation_svd_sqrt, axis=0))
    color_correlation_normalized = tensor(color_correlation_svd_sqrt / max_norm_svd_sqrt)
    return color_correlation_normalized

def lucid_colorspace_to_rgb(t,device = 'cuda'):

    t_flat = t.permute(0,2,3,1)
    # t_flat = torch.matmul(t_flat, color_correlation_normalized().T)
    t_flat = torch.matmul(t_flat , color_correlation_normalized().T.to(device))
    t = t_flat.permute(0,3,1,2)
    return t

def rgb_to_lucid_colorspace(t, device = 'cuda'):
    t_flat = t.permute(0,2,3,1)
    inverse = torch.inverse(color_correlation_normalized().T.to(device))
    t_flat = torch.matmul(t_flat, inverse)
    t = t_flat.permute(0,3,1,2)
    return t

def imagenet_mean_std(device = 'cuda'):
    return (tensor([0.485, 0.456, 0.406]).to(device), 
            tensor([0.229, 0.224, 0.225]).to(device))

def denormalize(x):
    mean, std = imagenet_mean_std()
    return x.float()*std[...,None,None] + mean[...,None,None]

def normalize(x, device = 'cuda'):
    mean, std = imagenet_mean_std(device = device)
    return (x-mean[...,None,None]) / std[...,None,None]

def image_buf_to_rgb(h, w, img_buf, device = 'cuda'):
    img = img_buf.detach()
    img = fft_to_rgb(h, w, img, device = device)
    img = lucid_colorspace_to_rgb(img, device=  device)
    img = torch.sigmoid(img)
    img = img[0]    
    return img
    
def show_rgb(img, label=None, ax=None, dpi=25, **kwargs):
    plt_show = True if ax == None else False
    if ax == None: _, ax = plt.subplots(figsize=(img.shape[2]/dpi,img.shape[1]/dpi))
    x = img.cpu().permute(1,2,0).numpy()
    ax.imshow(x)
    ax.axis('off')
    ax.set_title(label)
    if plt_show: plt.show()

def gpu_affine_grid(size):
    size = ((1,)+size)
    N, C, H, W = size
    grid = torch.FloatTensor(N, H, W, 2).cuda()
    linear_points = torch.linspace(-1, 1, W) if W > 1 else tensor([-1.])
    grid[:, :, :, 0] = torch.ger(torch.ones(H), linear_points).expand_as(grid[:, :, :, 0])
    linear_points = torch.linspace(-1, 1, H) if H > 1 else tensor([-1.])
    grid[:, :, :, 1] = torch.ger(linear_points, torch.ones(W)).expand_as(grid[:, :, :, 1])
    return vision.FlowField(size[2:], grid)

def tensor_stats(t, label=""):
    if len(label) > 0: label += " "
    return("%smean:%.2f std:%.2f max:%.2f min:%.2f" % (label, t.mean().item(),t.std().item(),t.max().item(),t.min().item()))