import torch
from ..auto_image_param import BaseImageParam
from .magnitude_spectrum import MagnitudeSpectrum
from ..utils import fft_to_rgb, lucid_colorspace_to_rgb, normalize, init_image_param

class MagnitudeConstrainedImageParam(BaseImageParam):
    def __init__(
        self, 
        height: int, 
        width: int,
        magnitude_spectrum: MagnitudeSpectrum, 
        device: str, 
        standard_deviation: float, 
        batch_size: int = 1
    ):
        super().__init__()
        assert height == magnitude_spectrum.height
        assert width == magnitude_spectrum.width

        param = init_image_param(
            height=height,
            width=width,
            sd=standard_deviation,
            device=device
        )

        # 2. Decouple the tensor into phase and amplitude
        amplitude, phase = torch.abs(param), torch.angle(param)

        # 3. Set the phase spectrum to be trainable
        self.param = phase.requires_grad_()

        # 4. Hardcode the amplitude values to be magnitude_spectrum.data
        self.amplitude_spectrum = magnitude_spectrum.data

        self.height = height
        self.width = width
        self.device = device
        self.batch_size = batch_size
        # Assuming an optimizer is set up later with self.param as the parameter to optimize

    def get_image_parameter(self):
        """
        Compute image param from the self.param and self.amplitude_spectrum.
        Here we recombine the amplitude and phase into a complex tensor,
        then perform an inverse FFT to get the spatial domain representation.
        """
        # Convert amplitude and phase back to a complex tensor
        complex_spectrum = self.amplitude_spectrum * torch.exp(1j * self.param)

        # Inverse FFT to go from frequency to spatial domain
        img_spatial = torch.fft.ifft2(complex_spectrum).real  # Taking the real part if necessary

        return img_spatial


    def postprocess(self, device):
        img = fft_to_rgb(
            height=self.height,
            width=self.width,
            image_parameter=self.get_image_parameter(),
            device=device,
        )
        img = lucid_colorspace_to_rgb(t=img, device=device)
        img = torch.sigmoid(img)
        return img

    def normalize(self, x, device):
        return normalize(x=x, device=device)

    def forward(self, device):
        if self.batch_size == 1:
            return self.normalize(self.postprocess(device=device), device=device)
        else:
            return torch.cat(
                [
                    self.normalize(self.postprocess(device=device), device=device)
                    for i in range(self.batch_size)
                ],
                dim=0,
            )
