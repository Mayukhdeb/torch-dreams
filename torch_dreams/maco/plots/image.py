from math import ceil
import torch
import numpy as np
from matplotlib import pyplot as plt
from ..types import Optional, Union


def _clip_percentile(tensor: Union[torch.Tensor, np.ndarray],
                     percentile: float) -> np.ndarray:
    """
    Apply clip according to percentile value (percentile, 100-percentile) of a tensor
    only if percentile is not None.

    Parameters
    ----------
    tensor
        tensor to clip.

    Returns
    -------
    tensor_clipped
        Tensor clipped accordingly to the percentile value.
    """

    assert 0. <= percentile <= 100., "Percentile value should be in [0, 100]"

    if percentile is not None:
        clip_min = np.percentile(tensor, percentile)
        clip_max = np.percentile(tensor, 100. - percentile)
        tensor = np.clip(tensor, clip_min, clip_max)

    return tensor


def _normalize(image: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    """
    Normalize an image in [0, 1].

    Parameters
    ----------
    image
        Image to prepare.

    Returns
    -------
    image
        Image ready to be used with matplotlib (in range[0, 1]).
    """
    image = np.array(image, np.float32)

    image -= image.min()
    image /= image.max()

    return image




def plot_maco(image, alpha, percentile_image=1.0, percentile_alpha=80):
    """
    Plot maco feature visualization image (take care of merging the alpha).

    Parameters
    ----------
    image
        Image to plot.
    alpha
        Alpha channel to plot.
    percentile_image
        Percentile value to use to ceil the image and avoid extreme values.
    percentile_alpha
        Percentile value to use to ceil the alpha channel. A higher value will result in a more
        transparent image with only the most important features.
    """

    image = np.array(image).copy()
    image = _clip_percentile(image, percentile_image)

    alpha = np.mean(np.array(alpha).copy(), -1, keepdims=True)
    alpha = np.clip(alpha, 0, np.percentile(alpha, percentile_alpha))
    alpha = alpha / alpha.max()

    image = image * alpha
    image = _normalize(image)

    plt.imshow(np.concatenate([image, alpha], -1))
    plt.axis('off')