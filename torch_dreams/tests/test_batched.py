import torchvision.models as models
from torch_dreams import Dreamer

from torch_dreams.auto_image_param import AutoImageParam
from torch_dreams.batched_image_param import BatchedAutoImageParam
from torch_dreams.batched_objective import BatchedObjective
import numpy as np
import torch
import os

import pytest


def make_custom_func(layer_number=0, channel_number=0):
    def custom_func(layer_outputs):
        loss = layer_outputs[layer_number][channel_number].mean()
        return -loss

    return custom_func


@pytest.mark.parametrize("iters", [1, 2, 10, 20])
@pytest.mark.parametrize("batch_size", [1, 2, 5])
def test_batched_auto_image_param(iters, batch_size):

    model = models.inception_v3(pretrained=True)
    dreamy_boi = Dreamer(model=model, device="cpu", quiet=False)

    image_param = BatchedAutoImageParam(batch_size=batch_size, device="cpu")

    result = dreamy_boi.render(
        layers=[model.Mixed_6a], iters=iters, image_parameter=image_param
    )

    for image_param in result.image_params:
        assert isinstance(
            image_param, AutoImageParam
        ), "should be an instance of auto_image_param"
        assert isinstance(image_param.__array__(), np.ndarray)
        assert isinstance(
            image_param.to_hwc_tensor(), torch.Tensor
        ), "should be a torch.Tensor"
        assert isinstance(
            image_param.to_chw_tensor(), torch.Tensor
        ), "should be a torch.Tensor"

    for i in range(batch_size):
        filename = f"test_single_model_iters_{iters}_batch_idx{i}.jpg"
        result[i].save(filename=filename)
        assert os.path.exists(filename)
        os.remove(filename)


@pytest.mark.parametrize("iters", [1, 2, 10, 20])
@pytest.mark.parametrize("batch_size", [1, 2, 5])
def test_batched_auto_image_param_with_custom_func(iters, batch_size):

    model = models.inception_v3(pretrained=True)
    dreamy_boi = Dreamer(model=model, device="cpu", quiet=False)

    image_param = BatchedAutoImageParam(batch_size=batch_size, device="cpu")

    objective_functions = []
    for i in range(batch_size):
        objective_functions.append(make_custom_func(layer_number=0, channel_number=i))

    batched_objective = BatchedObjective(objectives=objective_functions)

    result = dreamy_boi.render(
        layers=[model.Mixed_6a],
        iters=iters,
        image_parameter=image_param,
        custom_func=batched_objective,
    )

    for image_param in result.image_params:
        assert isinstance(
            image_param, AutoImageParam
        ), "should be an instance of auto_image_param"
        assert isinstance(image_param.__array__(), np.ndarray)
        assert isinstance(
            image_param.to_hwc_tensor(), torch.Tensor
        ), "should be a torch.Tensor"
        assert isinstance(
            image_param.to_chw_tensor(), torch.Tensor
        ), "should be a torch.Tensor"
