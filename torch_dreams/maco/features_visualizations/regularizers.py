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






def l_inf_reg(factor: float = 1.0): 
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
        max_abs_values = torch.max(torch.abs(images).flatten(start_dim=1), dim=1).values
        return factor * max_abs_values
    return l_inf




def total_variation_reg(factor: float = 1.0): 

    """
    Total variation regularization.

    Parameters
    ----------
    factor : float
        Weight that controls the importance of the regularization term.

    Returns
    -------
    Callable
        Function that computes the total variation of the images.
    """
    def tv_reg(images: torch.Tensor) -> torch.Tensor:
        # Compute the total variation by summing absolute differences without normalizing
        pixel_dif1 = torch.sum(torch.abs(images[:, :, 1:, :] - images[:, :, :-1, :]), dim=[1,2,3])
        pixel_dif2 = torch.sum(torch.abs(images[:, :, :, 1:] - images[:, :, :, :-1]), dim=[1,2,3])
        return factor * (pixel_dif1 + pixel_dif2)

    return tv_reg



