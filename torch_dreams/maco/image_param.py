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

        image_param = init_image_param(
            height=height,
            width=width,
            sd=standard_deviation,
            device=device
        )

        """
        torch.angle(image_param) would contain only 0 or pi. nothing else. this is because image_param contains only real values
        """
        # 2. Decouple the tensor into phase and amplitude
        amplitude, phase = torch.abs(image_param), torch.angle(image_param)
        
        # 3. Set the phase spectrum to be trainable
        self.param = phase.requires_grad_()

        # 4. Hardcode the amplitude values to be magnitude_spectrum.data
        # self.amplitude_spectrum = magnitude_spectrum.data.requires_grad_(False)
        self.amplitude_spectrum = amplitude

        self.height = height
        self.width = width
        self.device = device
        self.batch_size = batch_size
        
        """
        we should check whether we can recover image_param given phase and amplitude
        """
        self.check_whether_we_get_image_param(
            amplitude=amplitude,
            phase=phase,
            image_param=image_param
        )
        
    def reconstruct_image_param(self, amplitude, phase):
        reconstructed_image_param = amplitude * torch.exp(1j * phase)
        return reconstructed_image_param.real
    
    def check_whether_we_get_image_param(self, amplitude, phase, image_param):
        reconstructed_image_param = self.reconstruct_image_param(
            amplitude=amplitude,
            phase=phase
        )

        assert torch.allclose(
            image_param,
            reconstructed_image_param
        ), f"Could not reconstruct image param. Very sad."


    def postprocess(self, device):
        img = fft_to_rgb(
            height=self.height,
            width=self.width,
            image_parameter=self.reconstruct_image_param(amplitude = self.amplitude_spectrum, phase=self.param),
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
