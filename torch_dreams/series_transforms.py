# source: https://pytorch.org/vision/main/_modules/torchvision/transforms/transforms.html#RandomAffine
import numbers
from collections.abc import Sequence

import torch


class RandomSeriesTranslate(torch.nn.Module):
    """Random translation of the series.

    The series is expected to be a torch Tensor and have [N, C, L] shape.

    Args:
        translate (float): maximum absolute fraction for translation. Shift is randomly sampled
         in the range -length * translate < shift < length * translate.
        fill (series, number or None): Fill values for the translated series. Default is ``0``.
          If given a number, the value is used for all channels respectively.
    """

    def __init__(
        self,
        translate: float,
        fill=0,
    ):
        super().__init__()

        if not isinstance(translate, numbers.Number):
            raise TypeError(f"Fill should be a number but is {type(translate)}.")
            if not (0.0 <= translate <= 1.0):
                raise ValueError("translation value should be between 0 and 1")
        self.translate = translate

        if fill is None:
            fill = 0
        elif not isinstance(fill, (Sequence, numbers.Number)):
            raise TypeError("Fill should be either a sequence or a number.")

        self.fill = fill

    @staticmethod
    def get_params(
        translate,
        length,
    ):
        """Get parameters for translation

        Returns:
            absolute shift
        """
        max_shift = float(translate * length)
        shift = int(round(torch.empty(1).uniform_(-max_shift, max_shift).item()))
        return shift


    def forward(self, series):
        """
            series (torch.Tensor): Series to be transformed.

        Returns:
            torch.Tensor: translated series.
        """
        fill = self.fill
        channels, length = _get_series_dimensions(series)

        if fill is not None and not isinstance(fill, (float, int)):
            fill = torch.FloatTensor(fill)

        shift = self.get_params(self.translate, length)

        out = torch.roll(series, shift, dims=-1)

        if fill is not None and shift > 0:
            out[..., :shift] = fill

        if fill is not None and shift < 0:
            out[..., -shift:] = fill

        return out

    def __repr__(self) -> str:
        s = f"{self.__class__.__name__}(degrees={self.degrees}"
        s += f", translate={self.translate}" if self.translate is not None else ""
        s += f", fill={self.fill}" if self.fill != 0 else ""
        s += ")"

        return s


def _check_sequence_input(x, name, req_sizes):
    msg = req_sizes[0] if len(req_sizes) < 2 else " or ".join([str(s) for s in req_sizes])
    if not isinstance(x, Sequence):
        raise TypeError(f"{name} should be a sequence of length {msg}.")
    if len(x) not in req_sizes:
        raise ValueError(f"{name} should be a sequence of length {msg}.")

def _setup_angle(x, name, req_sizes=(2,)):
    if isinstance(x, numbers.Number):
        if x < 0:
            raise ValueError(f"If {name} is a single number, it must be positive.")
        x = [-x, x]
    else:
        _check_sequence_input(x, name, req_sizes)

    return [float(d) for d in x]


def _get_series_dimensions(series):
    # TODO:  _assert_image_tensor(img)
    if series.ndim == 1:
        channels = 1
        length = len(series)
    else:
        channels = series.shape[-2]
        length = series.shape[-1]
    return [channels, length]
