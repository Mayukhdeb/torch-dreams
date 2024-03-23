import torch

import numpy as np
from ..types import Optional, Union, Callable, Tuple
from .preconditioning import maco_image_parametrization, init_maco_buffer
from .objectives import Objective

import torchvision.transforms.functional as TF





def maco(objective: Objective,
         optimizer: Optional[torch.optim.Optimizer] = None,
         nb_steps: int = 256,
         noise_intensity: Optional[Union[float, Callable]] = 0.08,
         box_size: Optional[Union[float, Callable]] = None,
         nb_crops: Optional[int] = 32,
         values_range: Tuple[float, float] = (-1, 1),
         custom_shape: Optional[Tuple] = (512, 512)) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Optimise a single objective using MaCo method. Note that, unlike classic fourier optimization,
    we can only optimize for one objective at a time.

    Ref. Unlocking Feature Visualization for Deeper Networks with MAgnitude Constrained
         Optimization (2023).
    https://arxiv.org/abs/2306.06805

    Parameters
    ----------
    objective
        Objective object.
    optimizer
        Optimizer used for gradient ascent, default Nadam(lr=1.0).
    nb_steps
        Number of iterations.
    noise_intensity
        Control the noise injected at each step. Either a float : each step we add noise
        with same std, or a function that associate for each step a noise intensity.
    box_size
        Control the average size of the crop at each step. Either a fixed float (e.g 0.5 means
        the crops will be 50% of the image size) or a function that take as parameter the step
        and return the average box size. Default to linear decay from 50% to 5%.
    nb_crops
        Number of crops used at each steps, higher make the optimisation slower but
        make the results more stable. Default to 32.
    values_range
        Range of values of the inputs that will be provided to the model, e.g (0, 1) or (-1, 1).
    custom_shape
        If specified, optimizes images of the given size. Used with
        a low box size to optimize bigger images crop by crop.

    Returns
    -------
    image_optimized
        Optimized image for the given objective.
    transparency
        Transparency of the image, i.e the sum of the absolute value of the gradients
        of the image with respect to the objective.
    """
    values_range = (min(values_range), max(values_range))


    model, objective_function, _, input_shape = objective.compile()

    assert input_shape[0] == 1, "You can only optimize one objective at a time with MaCo."

    if optimizer is None:
        optimizer = torch.optim.NAdam(1.0)

    if box_size is None:
        # default to box_size that go from 50% to 5%
        box_size_values = torch.linspace(0.5, 0.05, nb_steps)
        get_box_size = lambda step_i: box_size_values[step_i]
    elif hasattr(box_size, "__call__"):
        get_box_size = box_size
    elif isinstance(box_size, float):
        get_box_size = lambda _ : box_size
    else:
        raise ValueError('box_size must be a function or a float.')
    

    if noise_intensity is None:
        # default to large noise to low noise
        noise_values = torch.logspace(0, -4, nb_steps)
        get_noise_intensity = lambda step_i: noise_values[step_i]
    elif hasattr(noise_intensity, "__call__"):
        get_noise_intensity = noise_intensity
    elif isinstance(noise_intensity, float):
        get_noise_intensity = lambda _ : noise_intensity
    else:
        raise ValueError('noise_intensity size must be a function or a float.')
    
    img_shape = (input_shape[1], input_shape[2])
    if custom_shape:
        img_shape = custom_shape

    magnitude, phase = init_maco_buffer(img_shape)
    phase = torch.nn.Parameter(phase)

    transparency = torch.zeros((*custom_shape, 3))

    for step_i in range(nb_steps):
        box_size_at_i = get_box_size(step_i)
        noise_intensity_at_i = get_noise_intensity(step_i)

        grads, grads_img = maco_optimisation_step(model, objective_function, magnitude, phase,
                                                  box_size_at_i, noise_intensity_at_i, nb_crops,
                                                  values_range)

        optimizer.zero_grad()
        grads.backward()
        optimizer.step()
        transparency += torch.abs(grads_img)

    img = maco_image_parametrization(magnitude, phase, values_range)



    return img, transparency







def maco_optimisation_step(model, objective_function, magnitude, phase,
                           box_average_size, noise_std, nb_crops, values_range):
    """Optimisation step for MaCo method in PyTorch.
    
    Parameters are the same as the TensorFlow version, adjusted for PyTorch usage.
    """
    # Ensure that `phase` requires gradient
    phase.requires_grad_(True)

    # Generate image from magnitude and phase
    image = maco_image_parametrization(magnitude, phase, values_range)

    # Initialize random crop parameters
    center_x = 0.5 + torch.randn((nb_crops,)) * 0.15
    center_y = 0.5 + torch.randn((nb_crops,)) * 0.15
    delta_x = torch.randn((nb_crops,)) * 0.05 + box_average_size
    delta_x = torch.clamp(delta_x, 0.05, 1.0)
    delta_y = delta_x  # Assume square boxes

    # Sample random crops from the image
    crops = []
    for i in range(nb_crops):
        crop = TF.resized_crop(image, int(center_y[i] - delta_y[i] * 0.5), int(center_x[i] - delta_x[i] * 0.5), 
                               int(delta_y[i]), int(delta_x[i]), model.input_shape[2:])
        crops.append(crop)
    crops = torch.stack(crops)

    # Add random noise for robustness
    crops += torch.randn_like(crops) * noise_std
    crops += (torch.rand_like(crops) * noise_std) - (noise_std / 2.0)

    # Forward pass through the model
    model_outputs = model(crops)
    loss = objective_function(model_outputs).mean()

    # Compute gradients
    loss.backward()
    grads_phase = phase.grad
    # In PyTorch, to get the gradient with respect to the image,
    # you must ensure the image was created with `requires_grad=True`
    grads_image = image.grad if image.requires_grad else None

    return grads_phase, grads_image


        



    


    