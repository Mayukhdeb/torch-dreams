import numbers
from collections.abc import Sequence

import torch


class RandomSeriesTranslate(torch.nn.Module):

    def __init__(
        self,
        translate: float,
        fill=0,
        seed=42,
    ):
        super().__init__()

        if not isinstance(translate, numbers.Number):
            raise TypeError(f"translate should be a number but is {type(translate)}.")
            if not (0.0 <= translate <= 1.0):
                raise ValueError("translation value should be between 0 and 1")
        self.translate = translate

        if fill is None:
            fill = 0
        elif not isinstance(fill, (Sequence, numbers.Number)):
            raise TypeError("Fill should be either a sequence or a number.")

        self.fill = fill
        
        self.seed = seed
        self.generator = torch.Generator()
        self.generator.manual_seed(seed)
        

    def forward(self, series):
        fill = self.fill
        channels, length = _get_series_dimensions(series)

        max_shift = float(self.translate * length)
        shift = int(round(torch.empty(1).uniform_(-max_shift, max_shift, generator=self.generator).item()))

        out = torch.roll(series, shift, dims=-1)

        if fill is not None and not isinstance(fill, (float, int)):
            fill = torch.FloatTensor(fill)[:shift]

        if fill is not None and shift > 0:
            out[..., :shift] = fill

        if fill is not None and shift < 0:
            out[..., -shift:] = fill

        return out


class RandomSeriesScale(torch.nn.Module):

    def __init__(
        self,
        min_scale: float,
        max_scale: float,
        seed=42,
    ):
        super().__init__()

        if not isinstance(min_scale, numbers.Number):
            raise TypeError(f"min_scale should be a number but is {type(min_scale)}.")
        if not isinstance(max_scale, numbers.Number):
            raise TypeError(f"max_scale should be a number but is {type(max_scale)}.")

        self.min_scale = min_scale
        self.max_scale = max_scale

        self.seed = seed
        self.generator = torch.Generator()
        self.generator.manual_seed(seed)

    def forward(self, series):
        scale = torch.empty(1).uniform_(self.min_scale, self.max_scale, generator=self.generator).to(series.device)
        out = series * scale
        return out


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
