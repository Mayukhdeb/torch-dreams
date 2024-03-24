import torch
from torchvision.transforms import RandomCrop,Resize,RandomHorizontalFlip,RandomVerticalFlip,Pad,Compose
from ..types import Tuple,Callable,List






def random_blur(sigma_range: Tuple[float, float] = (1.0, 2.0),
                kernel_size: int = 10) -> Callable:
    
    """
    Generate a function that apply a random gaussian blur to the batch.

    Parameters
    ----------
    sigma_range
        Min and max sigma (or scale) of the gaussian kernel.
    kernel_size
        Size of the gaussian kernel

    Returns
    -------
    blur
        Transformation function applying random blur.
    
    """
    uniform = torch.linspace(-(kernel_size - 1) / 2., (kernel_size - 1) / 2., kernel_size)
    uniform_xx, uniform_yy = torch.meshgrid(uniform, uniform, indexing="ij")

    sigma_min = max(sigma_range[0], 0.1)
    sigma_max = max(sigma_range[1], 0.1)

    def blur(batch: torch.Tensor) -> torch.Tensor:
        # Dynamically determine the output size from the input batch
        output_size = (batch.size(2), batch.size(3))

        sigma = torch.rand(1) * (sigma_max - sigma_min) + sigma_min
        kernel = torch.exp(-0.5 * (uniform_xx ** 2 + uniform_yy ** 2) / sigma ** 2)
        kernel /= kernel.sum()
        kernel = kernel.repeat(batch.shape[1], 1, 1, 1)

        if batch.dim() == 3:
            batch = batch.unsqueeze(0)

        padding = (kernel_size - 1) // 2
        pad = torch.nn.ReflectionPad2d(padding)
        batch = pad(batch)

        blurred = torch.nn.functional.conv2d(batch, kernel, groups=batch.shape[1])
        blurred = torch.nn.functional.interpolate(blurred, size=output_size)

        return blurred

    return blur




def random_jitter(delta: int=6) -> Callable:
    """
    Generate a function that apply a random jitter to the batch of images.

    Parameters
    ----------
    delta
        Max of the shift

    Returns
    -------
    jitter
        Transformation function applying random jitter.
    """

    def jitter(images:torch.Tensor) -> torch.Tensor:
        crop = RandomCrop((images.shape[-1] - delta, images.shape[-2] - delta))
        return crop(images)
    
    return jitter




def random_scale(scale_range: Tuple[float, float] = (0.95, 1.05)) -> Callable:
    """
    Generate a function that applies a random scale to a batch of images, preserving the aspect ratio.

    Parameters
    ----------
    scale_range : Tuple[float, float]
        Min and max scale factor.

    Returns
    -------
    Callable
        A transformation function applying random scale.
    """

    min_scale, max_scale = scale_range

    def scale(images: torch.Tensor) -> torch.Tensor:
        _, _, h, w = images.shape
        scale_factor = torch.empty(1).uniform_(min_scale, max_scale).item()
        new_h, new_w = int(h * scale_factor), int(w * scale_factor)

        # Resize the images using interpolate
        scaled_images = torch.nn.functional.interpolate(images, size=(new_h, new_w), mode='bilinear', align_corners=False)
        return scaled_images

    return scale



def random_flip(horizontal: bool = True, vertical: bool = True) -> Callable:
    """
    Generate a function that apply a random flip to the batch of images.

    Parameters
    ----------
    horizontal
        Apply horizontal flip.
    vertical
        Apply vertical flip.

    Returns
    -------
    flip
        Transformation function applying random flip.
    """
    def flip(images:torch.Tensor) -> torch.Tensor:
        if horizontal:
            if torch.rand(1) > 0.5:
                images = RandomHorizontalFlip(p=1)(images)
        if vertical:
            if torch.rand(1) > 0.5:
                images = RandomVerticalFlip(p=1)(images)
        return images
    
    return flip



def pad(size: int = 6,pad_value: float = 0) -> Callable:
    """
    Generate a function that apply a padding to the batch of images.

    Parameters
    ----------
    size
        Size of the padding.
    pad_value
        Value of the padding.

    Returns
    -------
    pad
        Transformation function applying padding.
    """
    def pad_fn(images:torch.Tensor) -> torch.Tensor:
        padding = Pad(padding = size,fill=pad_value)
        return padding(images)
    
    return pad_fn



def  compose_transformations(transformations: List[Callable]) -> Callable:
    """
    Generate a function that apply a list of transformations to the batch of images.

    Parameters
    ----------
    transformations
        List of transformations to apply.

    Returns
    -------
    composed
        Transformation function applying a list of transformations.
    """
    def composed(images:torch.Tensor) -> torch.Tensor:
        composed_transform = Compose(transformations)
        return composed_transform(images)
    
    return composed


def generate_standard_transformations(size: int) -> Callable:
    """
    Generate a function that apply a list of standard transformations to the batch of images.

    Parameters
    ----------
    size
        Size of the image

    Returns
    -------
    transformations
        A combinations of transformations to make the optimization robust.

    
    """
    unit = int(size/16)

    return compose_transformations([
        pad(unit * 4, 0.0),
        random_jitter(unit * 2),
        random_jitter(unit * 2),
        random_jitter(unit * 4),
        random_jitter(unit * 4),
        random_jitter(unit * 4),
        random_scale((0.92,0.96)),
        random_blur(sigma_range=(1.0,1.1)),
        random_jitter(unit),
        random_jitter(unit),
        random_flip()
    ])
    



