"""
Image Regularizers

"""

import torch
from ..types import Callable


def l1_reg(factor: float = 1.0) -> Callable:
    """
    Mean L1 regularization.

    Parameters
    ----------
    factor
        Weight that controls the importance of the regularization term.

    Returns
    -------
    reg
        Mean L1 of the images.
    """
    def reg(images: torch.Tensor) -> torch.Tensor:
        return factor * torch.mean(torch.abs(images), dim=(1, 2, 3))
    return reg


def l2_reg(factor: float = 1.0) -> Callable:
    """
    Mean L2 regularization.

    Parameters
    ----------
    factor
        Weight that controls the importance of the regularization term.

    Returns
    -------
    reg
        Mean L2 of the images.
    """
    def reg(images: torch.Tensor) -> torch.Tensor:
        return factor * torch.sqrt(torch.mean(images ** 2, dim=(1, 2, 3)))
    return reg



def l_inf_reg(factor: float = 1.0) -> Callable:
    """
    Mean L-inf regularization.

    Parameters
    ----------
    factor
        Weight that controls the importance of the regularization term.

    Returns
    -------
    l_inf
        Max of the images.
    """
    def l_inf(images: torch.Tensor) -> torch.Tensor:
        return factor * torch.max(torch.abs(images), dim=(1, 2, 3)).values
    return l_inf



def total_variation_reg(factor: float = 1.0) -> Callable:
    """
    Total variation regularization.

    Parameters
    ----------
    factor
        Weight that controls the importance of the regularization term.

    Returns
    -------
    tv_reg
        Total variation of the images.
    """
    def tv_reg(images: torch.Tensor) -> torch.Tensor:
        if len(images.shape) == 4:
            pixel_diff1 = images[..., :-1, :] - images[..., 1:, :]
            pixel_diff2 = images[..., :, :-1] - images[..., :, 1:]
        elif len(images.shape) == 3:
            pixel_diff1 = images[..., :-1] - images[..., 1:]
            pixel_diff2 = images[..., :-1] - images[..., 1:]
        else:
            raise ValueError("Unsupported number of dimensions in input tensor.")
        total_var = torch.sum(torch.abs(pixel_diff1)) + torch.sum(torch.abs(pixel_diff2))
        return factor * total_var
    return tv_reg


