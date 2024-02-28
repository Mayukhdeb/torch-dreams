import os

import numpy as np
import pytest
import torch
import torchvision.transforms as transforms

from torch_dreams import Dreamer
from torch_dreams.auto_series_param import AutoSeriesParam
from torch_dreams.series_transforms import RandomSeriesTranslate
from torch_dreams.transforms import random_resize


class CNN1d(torch.nn.Module):
    def __init__(self, in_channels, out_features, channels=25):
        super(CNN1d, self).__init__()
        self.conv1 = torch.nn.Conv1d(in_channels, channels, kernel_size=5, stride=2, padding=1)
        self.conv2 = torch.nn.Conv1d(channels, channels, kernel_size=3, stride=2, padding=1)
        self.conv3 = torch.nn.Conv1d(channels, channels, kernel_size=3, stride=2, padding=1)
        self.flatten = torch.nn.Flatten()
        self.fc = torch.nn.LazyLinear(out_features)

    def forward(self, x):
        h1 = self.conv1(x).relu()
        h2 = self.conv2(h1).relu()
        h3 = self.conv3(h2).relu()
        return self.fc(self.flatten(h3))


@pytest.mark.parametrize("out_features", [3, 10, 21])
@pytest.mark.parametrize("sequence_length", [11, 20, 40, 99, 1000])
@pytest.mark.parametrize("channels", [1, 2, 10, 59])
@pytest.mark.parametrize("batch_size", [1, 16, 64, 256])
def test_cnn_model_outputs_correct_shape(sequence_length, channels, out_features, batch_size):
    model = CNN1d(in_channels=channels, out_features=out_features)
    x = torch.zeros((batch_size, channels, sequence_length))

    assert model(x).shape == (batch_size, out_features)


@pytest.mark.parametrize("iters", [1, 2, 10, 20])
@pytest.mark.parametrize("sequence_length", [11, 20, 40, 99, 1000])
@pytest.mark.parametrize("channels", [1, 2, 10, 59])
@pytest.mark.parametrize("batch_size", [1])
def test_auto_series_param(iters, sequence_length, channels, batch_size):
    model = CNN1d(in_channels=channels, out_features=10)

    # Prepare lazy modules.
    x = torch.zeros((batch_size, channels, sequence_length))
    y = model(x)

    series_param = AutoSeriesParam(
        length=sequence_length,
        channels=channels,
        device="cpu",
        standard_deviation=0.01,
        batch_size=batch_size,
    )

    translate = 0.1
    scale_max = 1.2
    scale_min = 0.5

    series_transforms = transforms.Compose(
        [
            RandomSeriesTranslate(translate),
        ]
    )

    dreamy_boi = Dreamer(model=model, device='cpu', quiet=False)
    dreamy_boi.set_custom_transforms(series_transforms)

    result = dreamy_boi.render(
        layers=[model.conv1],
        iters=iters,
        image_parameter=series_param,
    )

    assert isinstance(result, AutoSeriesParam), "should be an instance of auto_series_param"
    assert isinstance(result.__array__(), np.ndarray)
    assert isinstance(result.to_cl_tensor(), torch.Tensor), "should be a torch.Tensor"
    assert isinstance(result.to_lc_tensor(), torch.Tensor), "should be a torch.Tensor"
    assert result.to_cl_tensor().shape == x[0].shape

def test_auto_series_save(iters=2, sequence_length=40, channels=2, batch_size=1):
    model = CNN1d(in_channels=channels, out_features=10)

    # Prepare lazy modules.
    x = torch.zeros((batch_size, channels, sequence_length))
    y = model(x)

    series_param = AutoSeriesParam(
        length=sequence_length,
        channels=channels,
        device="cpu",
        standard_deviation=0.01,
    )

    translate = 0.1
    scale_max = 1.2
    scale_min = 0.5

    series_transforms = transforms.Compose(
        [
            RandomSeriesTranslate(translate),
        ]
    )

    dreamy_boi = Dreamer(model=model, device='cpu', quiet=False)
    dreamy_boi.set_custom_transforms(series_transforms)

    result = dreamy_boi.render(
        layers=[model.conv1],
        iters=iters,
        image_parameter=series_param,
    )

    filename = f"test_ts_single_model.jpg"
    result.save(filename=filename)
    assert os.path.exists(filename)
    os.remove(filename)
