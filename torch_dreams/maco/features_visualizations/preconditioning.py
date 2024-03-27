"""
Images preconditionners

Adaptation of the original Lucid library :
https://github.com/tensorflow/lucid/blob/master/lucid/optvis/param/color.py
Credit is due to the original Lucid authors.
"""

import torch
import torch.fft
import torch.nn.functional as F
import numpy as np
from torchvision.transforms import Resize
from urllib.request import urlretrieve
import os
from pathlib import Path
import requests


IMAGENET_SPECTRUM_URL = "https://storage.googleapis.com/serrelab/loupe/"\
                        "spectrums/imagenet_decorrelated.npy"







# Define the recorrelate_colors function for PyTorch
def recorrelate_colors(images: torch.Tensor) -> torch.Tensor:
    """
    Map uncorrelated colors to 'normal colors' by using empirical color
    correlation matrix of ImageNet (see https://distill.pub/2017/feature-visualization/)

    Parameters
    ----------
    images : torch.Tensor
        Input samples, with N number of samples, W & H the sample dimensions,
        and C the number of channels.

    Returns
    -------
    images : torch.Tensor
        Images recorrelated.

    """
    imagenet_color_correlation = torch.tensor(
        [[0.56282854, 0.58447580, 0.58447580],
         [0.19482528, 0.00000000, -0.19482528],
         [0.04329450, -0.10823626, 0.06494176]],
        dtype=torch.float32
    )
    images_flat = images.reshape(-1, 3)
    images_flat = torch.matmul(images_flat, imagenet_color_correlation)
    return  torch.reshape(images_flat, images.shape)







def to_valid_rgb(images: torch.Tensor, normalizer: str = 'sigmoid', values_range: tuple = (0, 1)) -> torch.Tensor:
    """

    Apply transformations to map tensors to valid rgb images.

    Parameters
    ----------
    images : torch.Tensor
        Input samples, with N number of samples, W & H the sample dimensions,
        and C the number of channels.
    normalizer : str
        Transformation to apply to map pixels in the range [0, 1]. Either 'clip' or 'sigmoid'.
    values_range : tuple
        Range of values of the inputs that will be provided to the model, e.g (0, 1) or (-1, 1).

    Returns
    -------
    images : torch.Tensor
        Images after correction
    
    """
    images = recorrelate_colors(images)
    if normalizer == 'sigmoid':
        images = torch.sigmoid(images)
    elif normalizer == 'clip':
        images = torch.clamp(images, values_range[0], values_range[1])
    else:
        raise ValueError(f"Invalid normalizer: {normalizer}")
    
    # Rescale according to value range, now correctly handling the reduction over dimensions
    images_flat = images.view(images.size(0), -1)  # Flatten all dimensions except the batch
    min_vals = images_flat.min(dim=1, keepdim=True)[0].view(images.size(0), 1, 1, 1)
    max_vals = images_flat.max(dim=1, keepdim=True)[0].view(images.size(0), 1, 1, 1)
    
    images = (images - min_vals) / (max_vals - min_vals)  # Normalize to [0, 1]
    images = images * (values_range[1] - values_range[0]) + values_range[0]  # Scale to [min_value, max_value]
    
    return images






def fft_2d_freq(width: int, height: int) -> np.ndarray:
    """
    Return the fft samples frequencies for a given width/height.
    As we deal with real values (pixels), the Discrete Fourier Transform is
    Hermitian symmetric, tensorflow's reverse operation requires only
    the unique components (width, height//2+1).

    Parameters
    ----------
    width
        Width of the image.
    height
        Height of the image.

    Returns
    -------
    frequencies
        Array containing the samples frequency bin centers in cycles per pixels
    """
    freq_y = np.fft.fftfreq(height)[:, np.newaxis]

    cut_off = int(width % 2 == 1)
    freq_x = np.fft.fftfreq(width)[:width//2+1+cut_off]

    return np.sqrt(freq_x**2 + freq_y**2)



def get_fft_scale(width: int, height: int, decay_power: float = 1.0) -> torch.Tensor:
    """
    Generate 'scaler' to normalize spectrum energy. Also scale the energy by the
    dimensions to use similar learning rate regardless of image size.
    adaptation of : https://github.com/tensorflow/lucid/blob/master/lucid/optvis/param/spatial.py
    #L73

    Parameters
    ----------
    width
        Width of the image.
    height
        Height of the image.
    decay_power
        Control the allowed energy of the high frequency, a high value
        suppresses high frequencies.

    Returns
    -------
    fft_scale
        Scale factor of the fft spectrum
    """
    frequencies = fft_2d_freq(width, height)
    fft_scale = 1.0 / np.maximum(frequencies, 1.0 / max(width, height)) ** decay_power
    fft_scale = fft_scale * np.sqrt(width * height)

    return torch.tensor(fft_scale, dtype=torch.complex64)




def fft_to_rgb(shape: tuple, buffer: torch.Tensor, fft_scale: torch.Tensor) -> torch.Tensor:
    """
    Convert a fft buffer into images.

    Parameters
    ----------
    shape : tuple
        Shape of the images with N number of samples, W & H the sample
        dimensions, and C the number of channels.
    buffer : torch.Tensor
        Image buffer in the fourier basis.
    fft_scale : torch.Tensor
        Scale factor of the fft spectrum

    Returns
    -------
    images : torch.Tensor
        Images in the 'pixels' basis.
    """
    batch, width, height, channels = shape
    spectrum = torch.complex(buffer[0],buffer[1]) * fft_scale

    image = torch.fft.irfft2(spectrum)
    image = image.permute(0, 2, 3, 1)
    image = image[:batch, :width, :height, :channels]

    return image / 4.0




def fft_image(shape: tuple, std: float = 0.01) -> torch.Tensor:
    """
    Generate the preconditioned image buffer

    Parameters
    ----------
    shape : tuple
        Shape of the images with N number of samples, W & H the sample
        dimensions, and C the number of channels.
    std : float
        Standard deviation of the normal for the buffer initialization

    Returns
    -------
    buffer : torch.Tensor
        Image buffer in the fourier basis.
    """
    batch, width, height, channels = shape
    frequencies = fft_2d_freq(width, height)

    buffer = torch.normal(0, std, (2, batch, channels)+frequencies.shape)

    return buffer



def download_file(url, file_path):
    """Downloads file from the url and saves it as file_path"""
    response = requests.get(url)
    response.raise_for_status()  # Check that the request was successful
    with open(file_path, 'wb') as f:
        f.write(response.content)



def get_file(filename, url, cache_subdir="spectrums"):
    """Replicates tf.keras.utils.get_file functionality"""
    cache_dir = Path.home() / ".cache" / cache_subdir
    cache_dir.mkdir(parents=True, exist_ok=True)
    file_path = cache_dir / filename
    if not file_path.exists():
        download_file(url, file_path)
    return str(file_path)





def init_maco_buffer(image_shape, std=1.0):
    '''
    Initialize the buffer for the MACO algorithm.

    Parameters
    ----------
    image_shape
        Shape of the images with N number of samples, W & H the sample
        dimensions, and C the number of channels.
    std
       Standard deviation of the normal for the buffer initialization

   Returns
   -------
   magnitude
       Magnitude of the spectrum
   phase
       Phase of the spectrum
    '''
    spectrum_shape = (image_shape[0], image_shape[1]//2+1)
    phase = np.random.normal(size=(3, *spectrum_shape), scale=std).astype(np.float32)
    magnitude_p = get_file("imagenet_decorrelated.npy", IMAGENET_SPECTRUM_URL)
    magnitude = np.load(magnitude_p)
    magnitude = np.moveaxis(magnitude, 0, -1)
    magnitude_resized = torch.nn.functional.interpolate(torch.tensor(magnitude).permute(2, 0, 1).unsqueeze(0), size=spectrum_shape, mode='bilinear', align_corners=False).squeeze(0).permute(1, 2, 0).numpy()
    
    magnitude = np.moveaxis(magnitude_resized, -1, 0)
    
    return torch.tensor(magnitude, dtype=torch.float32), torch.tensor(phase, dtype=torch.float32)
    



def maco_image_parametrization(magnitude, phase, values_range):
    """
    Generate the image from the magnitude and phase using MaCo method.

    Parameters
    ----------
    magnitude : torch.Tensor
        Magnitude of the spectrum.
    phase : torch.Tensor
        Phase of the spectrum.
    values_range : tuple
        Range of values of the inputs that will be provided to the model, e.g (0, 1) or (-1, 1).

    Returns
    -------
    img : torch.Tensor
        Image in the 'pixels' basis.
    """
    phase = phase - torch.mean(phase)
    phase = phase / (torch.std(phase) + 1e-5)

    buffer = torch.complex(magnitude * torch.cos(phase), magnitude * torch.sin(phase))

    img = torch.fft.irfft2(buffer)
    img = img.permute(1, 2, 0)

    img = img - torch.mean(img)
    img = img / (torch.std(img) + 1e-5)

    img = recorrelate_colors(img)  # Assuming you have a similar function in PyTorch
    img = torch.sigmoid(img)

    img = img * (values_range[1] - values_range[0]) + values_range[0]

    return img



   

