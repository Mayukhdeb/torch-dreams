import numpy as np
import torch
from torch import tensor
import matplotlib.pyplot as plt
from torchvision import transforms
import fastai.vision as vision

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

def get_fft_scale(h, w, decay_power=.75):
    d=.5**.5 # set center frequency scale to 1
    fy = np.fft.fftfreq(h,d=d)[:,None]
    if w % 2 == 1:
        fx = np.fft.fftfreq(w,d=d)[: w // 2 + 2]
    else:
        fx = np.fft.fftfreq(w,d=d)[: w // 2 + 1]        
    freqs = (fx*fx + fy*fy) ** decay_power
    scale = 1.0 / np.maximum(freqs, 1.0 / (max(w, h)*d))
    scale = tensor(scale).float()[None,None,...,None].cuda()
    return scale

def fft_to_rgb(height, width, image_parameter, **kwargs):
    scale = get_fft_scale(height, width, **kwargs)
    t = scale * image_parameter
    t = torch.irfft(t, 2, normalized=True, signal_sizes=(height,width))
    return t

def rgb_to_fft(h, w, t, **kwargs):
    t = torch.rfft(t, normalized=True, signal_ndim=2)
    scale = get_fft_scale(h, w, **kwargs)
    t = t / scale
    return t

def color_correlation_normalized():
    color_correlation_svd_sqrt = np.asarray([[0.26, 0.09, 0.02],
                                             [0.27, 0.00, -0.05],
                                             [0.27, -0.09, 0.03]]).astype(np.float32)
    max_norm_svd_sqrt = np.max(np.linalg.norm(color_correlation_svd_sqrt, axis=0))
    color_correlation_normalized = tensor(color_correlation_svd_sqrt / max_norm_svd_sqrt).cuda()
    return color_correlation_normalized

def lucid_colorspace_to_rgb(t):
    t_flat = t.permute(0,2,3,1)
    t_flat = torch.matmul(t_flat, color_correlation_normalized().T)
    t = t_flat.permute(0,3,1,2)
    return t

def rgb_to_lucid_colorspace(t):
    t_flat = t.permute(0,2,3,1)
    inverse = torch.inverse(color_correlation_normalized().T)
    t_flat = torch.matmul(t_flat, inverse)
    t = t_flat.permute(0,3,1,2)
    return t

def imagenet_mean_std():
    return (tensor([0.485, 0.456, 0.406]).cuda(), 
            tensor([0.229, 0.224, 0.225]).cuda())

def denormalize(x):
    mean, std = imagenet_mean_std()
    return x.float()*std[...,None,None] + mean[...,None,None]

def normalize(x):
    mean, std = imagenet_mean_std()
    return (x-mean[...,None,None]) / std[...,None,None]

def image_buf_to_rgb(h, w, img_buf, **kwargs):
    img = img_buf.detach()
    img = fft_to_rgb(h, w, img, **kwargs)
    img = lucid_colorspace_to_rgb(img)
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

# def lucid_transforms(img, jitter=None, scale=.5, degrees=45, **kwargs):
#     h,w = img.shape[-2], img.shape[-1]
#     if jitter is None:
#         jitter = min(h,w)//2
#     fastai_image = vision.Image(img.squeeze())

#     # pad
#     fastai_image._flow = gpu_affine_grid(fastai_image.shape)
#     vision.transform.pad()(fastai_image, jitter)

#     # jitter
#     first_jitter = int((jitter*(2/3)))
#     vision.transform.crop_pad()(fastai_image,
#                                 (h+first_jitter,w+first_jitter), 
#                                 row_pct=np.random.rand(), col_pct=np.random.rand())

#     # scale
#     percent = scale * 100 # scale up to integer to avoid float repr errors
#     scale_factors = [(100 - percent + percent/5. * i)/100 for i in range(11)]            
#     rand_scale = scale_factors[int(np.random.rand()*len(scale_factors))]
#     fastai_image._flow = gpu_affine_grid(fastai_image.shape)
#     vision.transform.zoom()(fastai_image, rand_scale)

#     # rotate
#     rotate_factors = list(range(-degrees, degrees+1)) + degrees//2 * [0]
#     rand_rotate = rotate_factors[int(np.random.rand()*len(rotate_factors))]
#     fastai_image._flow = gpu_affine_grid(fastai_image.shape)
#     vision.transform.rotate()(fastai_image, rand_rotate)

#     # jitter
#     vision.transform.crop_pad()(fastai_image, (h,w), row_pct=np.random.rand(), col_pct=np.random.rand())

    # return fastai_image.data[None,:]

def tensor_stats(t, label=""):
    if len(label) > 0: label += " "
    return("%smean:%.2f std:%.2f max:%.2f min:%.2f" % (label, t.mean().item(),t.std().item(),t.max().item(),t.min().item()))

def cossim(act0, act1, cosim_weight=0, **kwargs):
    dot = (act0 * act1).sum()
    mag0 = act0.pow(2).sum().sqrt()
    mag1 = act1.pow(2).sum().sqrt()
    cossim = cosim_weight*dot/(mag0*mag1)
    return cossim

def visualize_feature(model, layer, feature, start_image=None, last_hook_out=None,
                      size=200, steps=500, lr=0.004, weight_decay=0.1, grad_clip=1,
                      debug=False, frames=10, show=True, **kwargs):
    h,w = size if type(size) is tuple else (size,size)
    
    img_buf = init_fft_buf(h, w, **kwargs)
    img_buf.requires_grad_()
    opt = torch.optim.AdamW([img_buf], lr=lr, weight_decay=weight_decay)

    hook_out = None
    def callback(m, i, o):
        nonlocal hook_out
        hook_out = o
    hook = layer.register_forward_hook(callback)
    
    for i in range(1,steps+1):
        opt.zero_grad()
        
        img = fft_to_rgb(h, w, img_buf, **kwargs)
        img = lucid_colorspace_to_rgb(img)
        stats = tensor_stats(img)
        img = torch.sigmoid(img)
        img = normalize(img)
#         img = lucid_transforms(img, **kwargs)          
        model(img.cuda())        
        if feature is None:
            loss = -1 * hook_out[0].pow(2).mean()
        else:
            loss = -1 * hook_out[0][feature].mean()
        if last_hook_out is not None:
            simularity = cossim(hook_out[0], last_hook_out, **kwargs)
            loss = loss + loss * simularity

        loss.backward()
        torch.nn.utils.clip_grad_norm_(img_buf,grad_clip)
        opt.step()
        
        if debug and (i)%(int(steps/frames))==0:
            clear_output(wait=True)
            label = f"step: {i} loss: {loss:.2f} stats:{stats}"
            show_rgb(image_buf_to_rgb(h, w, img_buf, **kwargs),
                     label=label, **kwargs)

    hook.remove()
    
    retval = image_buf_to_rgb(h, w, img_buf, **kwargs)
    if show:
        if not debug: show_rgb(retval, **kwargs)
    return retval, hook_out[0].clone().detach()
