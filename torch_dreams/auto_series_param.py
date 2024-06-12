import torch

from .base_series_param import BaseSeriesParam
from .utils import init_series_param
from .utils import fft_to_series


class AutoSeriesParam(BaseSeriesParam):
    """Trainable series parameter which can be used to activate
       different parts of a neural net

    Args:
        length (int): The sequence length of the series
        channels (int): The number of channels of the series

        device (str): 'cpu' or 'cuda'
        standard_deviation (float): Standard deviation of the series initiated
         in the frequency domain.
        batch_size (int): The batch size of the input tensor. If batch_size=1,
         no batch dimension is expected.
    """

    def __init__(
            self,
            length,
            channels,
            device,
            standard_deviation,
            normalize_mean=None,
            normalize_std=None,
            batch_size: int = 1,
            seed: int = 42,
    ):
        # odd length is resized to even with one extra element
        if length % 2 == 1:
            param = init_series_param(
                batch_size=batch_size,
                channels=channels,
                length=length + 1,
                sd=standard_deviation,
                seed=seed,
                device=device,
            )
        else:
            param = init_series_param(
                batch_size=batch_size,
                channels=channels,
                length=length,
                sd=standard_deviation,
                seed=seed,
                device=device,
            )

        super().__init__(
            batch_size=batch_size,
            channels=channels,
            length=length,
            param=param,
            normalize_mean=normalize_mean,
            normalize_std=normalize_std,
            device=device,
        )

        self.standard_deviation = standard_deviation

    def postprocess(self, device):
        series = fft_to_series(
            channels=self.channels,
            length=self.length,
            series_parameter=self.param,
            device=device,
        )
        #TODO: img = lucid_colorspace_to_rgb(t=img, device=device)
        series = torch.sigmoid(series)
        return series
