import numpy as np
import torch
from torch import tensor
from torchvision import transforms

from .image_transforms import resize_4d_tensor_by_size
from .error_handlers import PytorchVersionError
from .constants import Constants

def check_pytorch_version():
    '''
    does not raise error if torch >=1.8.x
    else raises PytorchVersionError
    '''

    version = torch.__version__.split(".")
    main_version = int(version[0])

    if main_version < 1:
        PytorchVersionError(version=torch.__version__)
    elif main_version == 1:
        sub_version = int(version[1])
        if sub_version < 8:
            PytorchVersionError(version=torch.__version__)
        else:
            pass
    else:
        pass

def init_image_param(height, width, sd=0.01, device="cuda"):
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


def init_series_param(batch_size, channels, length, sd=0.01, seed=42, device="cuda"):
    """Initializes a series parameter in the frequency domain

    Args:
        batch_size (int): batch size of series
        channels (int): number of channels of series
        length (int): length of series
        sd (float, optional): Standard deviation of step values. Defaults to 0.01.
        device (str): 'cpu' or 'cuda'

    Returns:
        torch.tensor: series param to backpropagate on
    """
    np.random.seed(seed=seed)
    buffer = np.random.normal(size=(batch_size, channels, length), scale=sd).astype(np.float32)
    spectrum_t = tensor(buffer).float().to(device)
    return spectrum_t


def get_fft_scale(h, w, decay_power=0.75, device="cuda"):
    d = 0.5**0.5  # set center frequency scale to 1
    fy = np.fft.fftfreq(h, d=d)[:, None]

    if w % 2 == 1:
        fx = np.fft.rfftfreq(w, d=d)[: (w + 1) // 2]
    else:

        fx = np.fft.rfftfreq(w, d=d)[: w // 2]

    freqs = (fx * fx + fy * fy) ** decay_power
    scale = 1.0 / np.maximum(freqs, 1.0 / (max(w, h) * d))
    scale = tensor(scale).float().to(device)

    return scale


def get_fft_series_scale(length: int, decay_power: float = 0.75, device: str = "cuda"):
    d = 0.5**0.5  # set center frequency scale to 1

    if length % 2 == 1:
        fx = np.fft.rfftfreq(length, d=d)[: (length + 1) // 2]
    else:
        fx = np.fft.rfftfreq(length, d=d)[: length // 2]

    freqs = (fx * fx) ** decay_power

    scale = 1.0 / np.maximum(freqs, 1.0 / (length * d))
    scale = tensor(scale).float().to(device)

    return scale



def fft_to_rgb(height, width, image_parameter, device="cuda"):
    """convert image param to NCHW

    Latest docs: https://pytorch.org/docs/stable/fft.html

    Also refer:
        https://github.com/pytorch/pytorch/issues/49637

    Args:
        height (int): height of image
        width (int): width of image
        image_parameter (auto_image_param): auto_image_param.param

    Returns:
        torch.tensor: NCHW tensor

    """
    scale = get_fft_scale(height, width, device=device).to(image_parameter.device)
    # print(scale.shape, image_parameter.shape)
    if width % 2 == 1:
        image_parameter = image_parameter.reshape(1, 3, height, (width + 1) // 2, 2)
    else:
        image_parameter = image_parameter.reshape(1, 3, height, width // 2, 2)

    image_parameter = torch.complex(image_parameter[..., 0], image_parameter[..., 1])
    t = scale * image_parameter

    version = torch.__version__.split(".")[:2]
    main_version = int(version[0])
    sub_version = int(version[1])

    t = torch.fft.irfft2(t, s=(height, width), norm="ortho")

    return t


def fft_to_series(channels, length, series_parameter, device="cuda"):
    """convert series param to NCL

    WARNING: torch v1.7.0 works differently from torch v1.8.0 on fft.
    torch-dreams supports ONLY 1.8.x

    Latest docs: https://pytorch.org/docs/stable/fft.html

    Also refer:
        https://github.com/pytorch/pytorch/issues/49637

    Args:
        channels (int): number of channels of series
        length (int): length of series
        series_parameter (auto_series_param): auto_series_param.param

    Returns:
        torch.tensor: NCHW tensor

    """
    scale = get_fft_series_scale(length, device=device).to(series_parameter.device)

    if length % 2 == 1:
        series_parameter = series_parameter.reshape(1, channels, (length + 1) // 2, 2)
    else:
        series_parameter = series_parameter.reshape(1, channels, length // 2, 2)

    series_parameter = torch.complex(series_parameter[..., 0], series_parameter[..., 1])
    t = scale * series_parameter
    t = torch.fft.irfft(t, n=length, norm="ortho")

    return t


def lucid_colorspace_to_rgb(t, device="cuda"):

    t_flat = t.permute(0, 2, 3, 1)
    t_flat = torch.matmul(
        t_flat.to(device), Constants.color_correlation_matrix.T.to(device)
    )
    t = t_flat.permute(0, 3, 1, 2)
    return t


def normalize(x, device="cuda"):
    return (
        x - Constants.imagenet_mean[..., None, None].to(device)
    ) / Constants.imagenet_std[..., None, None].to(device)


def get_fft_scale_custom_img(h, w, decay_power=0.75, device="cuda"):
    d = 0.5**0.5  # set center frequency scale to 1
    fy = np.fft.fftfreq(h, d=d)[:, None]

    fx = np.fft.rfftfreq(w, d=d)[: (w // 2) + 1]
    freqs = (fx * fx + fy * fy) ** decay_power
    scale = 1.0 / np.maximum(freqs, 1.0 / (max(w, h) * d))
    scale = torch.tensor(scale).float().to(device)

    return scale


def get_fft_scale_custom_series(length, decay_power=0.75, device="cuda"):
    d = 0.5**0.5  # set center frequency scale to 1
    fx = np.fft.rfftfreq(length, d=d)[: (length // 2) + 1]
    freqs = (fx * fx) ** decay_power
    scale = 1.0 / np.maximum(freqs, 1.0 / (length * d))
    scale = torch.tensor(scale).float().to(device)

    return scale


def denormalize(x):

    return x.float() * Constants.imagenet_std[..., None, None].to(
        x.device
    ) + Constants.imagenet_mean[..., None, None].to(x.device)


def rgb_to_lucid_colorspace(t, device="cuda"):
    t_flat = t.permute(0, 2, 3, 1)
    inverse = torch.inverse(Constants.color_correlation_matrix.T.to(device))
    t_flat = torch.matmul(t_flat.to(device), inverse)
    t = t_flat.permute(0, 3, 1, 2)
    return t


def series_space_to_lucid_space(t, channel_correlation_matrix, device="cuda"):
    t_flat = t.permute(0, 2, 1)
    inverse = torch.inverse(channel_correlation_matrix.T.to(device))
    t_flat = torch.matmul(t_flat.to(device), inverse)
    t = t_flat.permute(0, 2, 1)
    return t


def chw_rgb_to_fft_param(x, device):
    im_tensor = torch.tensor(x).unsqueeze(0).float()

    x = rgb_to_lucid_colorspace(denormalize(im_tensor), device=device)

    x = torch.fft.rfft2(x, s=(x.shape[-2], x.shape[-1]), norm="ortho")
    return x


def cl_series_to_fft_param(x, channel_correlation_matrix, device):
    length = x.shape[-1]
    series_tensor = torch.tensor(x).float()

    x = series_space_to_lucid_space(
        series_tensor,
        channel_correlation_matrix=channel_correlation_matrix,
        device=device,
    )

    print(x.shape)

    x = torch.fft.rfft(x, n=length, norm="ortho")

    print(x.shape)

    return x


def fft_to_rgb_custom_img(height, width, image_parameter, device="cuda"):

    scale = get_fft_scale_custom_img(height, width, device=device).to(
        image_parameter.device
    )
    t = scale * image_parameter

    version = torch.__version__.split(".")[:2]
    main_version = int(version[0])
    sub_version = int(version[1])

    t = torch.fft.irfft2(t, s=(height, width), norm="ortho")

    return t
