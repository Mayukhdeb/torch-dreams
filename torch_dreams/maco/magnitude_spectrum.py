import torch
import numpy as np
from PIL import Image
from typing import List
from tqdm import tqdm

def get_magnitude_spectrum(image: Image, take_log = False):
    #Convert the image to grayscale
    image_gray = image.convert("L")
    
    # Convert the PIL image to a PyTorch tensor with values between 0 to 1
    image_tensor = torch.FloatTensor(np.array(image_gray)/255.)
    
    # Add a batch dimension and channel dimension to fit the FFT input requirements
    image_tensor = image_tensor.unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, H, W]
    
    # Use torch.fft.fftn for n-dimensional FFT, here applied to 2D (image)
    # The function requires a complex input, but our image is real, so we make it complex
    # with a zero imaginary part using torch.view_as_complex
    fft_result = torch.fft.fftn(image_tensor, dim=[2, 3])
    
    # Compute the power spectrum (magnitude squared of the FFT)
    # abs() computes the magnitude for complex tensors, then we square it
    power_spectrum = torch.abs(fft_result) ** 2
    
    # Shift the zero frequency component to the center
    power_spectrum_centered = torch.fft.fftshift(power_spectrum, dim=[2,3])
    
    # Removing the extra dimensions to visualize or process the power spectrum
    power_spectrum_centered = power_spectrum_centered.squeeze()

    if take_log:
        power_spectrum_centered = torch.log(
            1 + power_spectrum_centered
        )
    
    return power_spectrum_centered

def get_mean_magnitude_spectrum(
    images: List["Image"],
    take_log = False,
    progress = False
):  
    power_spectrums = []
    for image in tqdm(images, disable = not(progress)):
        ## chatgpt assert image shape ==  images[0] shape
        assert image.size == images[0].size, f"Please make sure that all of the images have the same height and width. I found a mismatch: {image.size} != {images[0].size}"
        power_spectrum = get_magnitude_spectrum(
            image=image,
            take_log=take_log
        )
        assert power_spectrum.ndim == 2, f"Expected only 2 dimensions in the power spectrum (height, width) but got {power_spectrum.ndim} dimensions"
        power_spectrums.append(power_spectrum.unsqueeze(0))

    mean_power_spectrum = torch.cat(
        power_spectrums,
        dim = 0
    ).mean(0)
    
    return mean_power_spectrum
