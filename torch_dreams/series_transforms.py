from __future__ import annotations

from numbers import Number
from collections.abc import Sequence

import torch


class RandomSeriesTranslate(torch.nn.Module):

    def __init__(
        self,
        translate: float,
        fill: Number | Sequence | None =0,
    ):
        super().__init__()

        if not isinstance(translate, Number):
            raise TypeError(f"translate should be a number but is {type(translate)}.")
        if not (0.0 <= translate <= 1.0):
            raise ValueError("translation value should be between 0 and 1")
        self.translate = translate

        if fill is not None and not isinstance(fill, (Sequence, Number)):
            raise TypeError("Fill must be either a sequence, a number, or None.")
        self.fill = fill

    def forward(self, series):
        fill = self.fill
        channels, length = _get_series_dimensions(series)

        max_shift = float(self.translate * length)
        shift = int(round(torch.empty(1).uniform_(-max_shift, max_shift).item()))

        # if fill is None, the overflow values are rolled over to the other end of the series
        out = torch.roll(series, shift, dims=-1)

        if fill is not None and not isinstance(fill, Number):
            # fill must be a sequence, check that length matches
            if shift > len(fill):
                raise RuntimeError(f'random shift greater than fill length ({shift} > {len(fill)})')

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
    ):
        super().__init__()

        if not isinstance(min_scale, Number):
            raise TypeError(f"min_scale should be a number but is {type(min_scale)}.")
        if not isinstance(max_scale, Number):
            raise TypeError(f"max_scale should be a number but is {type(max_scale)}.")

        self.min_scale = min_scale
        self.max_scale = max_scale

    def forward(self, series):
        scale = torch.empty(1).uniform_(self.min_scale, self.max_scale)
        out = series * scale
        return out


def _get_series_dimensions(series):
    # TODO:  _assert_image_tensor(img)
    if series.ndim == 1:
        channels = 1
        length = len(series)
    else:
        channels = series.shape[-2]
        length = series.shape[-1]
    return [channels, length]
