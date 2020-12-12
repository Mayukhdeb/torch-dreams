import torch
import numpy as np

from .utils import pytorch_input_adapter
from .utils import find_random_roll_values_for_tensor
from .utils import roll_torch_tensor
from .utils import get_random_rotation_angle
from .utils import rotate_image_tensor
from .utils import CascadeGaussianSmoothing

from .constants import UPPER_IMAGE_BOUND
from .constants import LOWER_IMAGE_BOUND
from .constants import UPPER_IMAGE_BOUND_GRAY
from .constants import LOWER_IMAGE_BOUND_GRAY

from .dreamer_utils import get_gradients

def dream_on_octave_with_masks(model, image_np, layers, iterations, lr,  custom_funcs = [None], max_rotation = 0.2, gradient_smoothing_coeff = None, gradient_smoothing_kernel_size = None, grad_mask =None, device = None, default_func = None):

    """
    Core function for image optimization with gradient masks 

    input args{
        model: pytorch model 
        image_np: 3 channel numpy image with shape (C,H,W)
        layers: list of layers whose outputs are to be used [model.layer1, model.layer2]
        lr: learning rate
        custom_funcs: list of custom functions to be applied with the corrresponding gradient masks 
        max_rotation: caps the max amount of rotation on the input tensor, helps reduce noise 
        gradient_smoothing_coeff: helps reduce fig frequency noise in gradients before adding
        gradient_smoothing_kernel_size: kernel size to be used while smoothing 
        grad_mask: list of gradient masks to be applied , make sure that len(grad_mask) = len(custom_funcs)
        device: cuda or CPU depending on the availability
        default_func: default func to be used if no custom_func is not given

    }
    """
        
    image_tensor = pytorch_input_adapter(image_np, device = device)
    if grad_mask is not None:
        grad_mask_tensors = [pytorch_input_adapter(g_mask, device = device).double() for g_mask in grad_mask]

    for i in range(iterations):
        """
        rolling 
        """
        roll_x, roll_y = find_random_roll_values_for_tensor(image_tensor)

        image_tensor_rolled = roll_torch_tensor(image_tensor, roll_x, roll_y) 
        
        """
        rotating
        """
        theta = get_random_rotation_angle(theta_max= max_rotation)

        image_tensor_rolled_rotated = rotate_image_tensor(image_tensor = image_tensor_rolled, theta = theta, device = device)

        """
        getting gradients
        """

        gradients_tensors = []
        for c in range(len(custom_funcs)):

            gradients_tensor = get_gradients(net_in = image_tensor_rolled_rotated, net = model, layers = layers,default_func = default_func,custom_func= custom_funcs[c]).detach()
            gradients_tensors.append(gradients_tensor)
        """
        unrotate and unroll gradients of the image tensor
        """
        gradients_tensors_unrotated  = [rotate_image_tensor(g, theta = -theta, device = device) for g in gradients_tensors]
        gradients_tensors = [roll_torch_tensor(g, -roll_x, -roll_y)  for g in gradients_tensors_unrotated]

        """
        image update
        """
        # print(gradient_smoothing_sigma, gradient_smoothing_kernel_size)
        
        if gradient_smoothing_kernel_size is not None and gradient_smoothing_coeff is not None:
            
            sigma = ((i + 1) / iterations) * 2.0 + gradient_smoothing_coeff
            gradients_tensors = [CascadeGaussianSmoothing(kernel_size = gradient_smoothing_kernel_size, sigma = sigma, device = device)(gradients_tensor.unsqueeze(0)).squeeze(0) for gradients_tensor in gradients_tensors]

            for m in range(len(gradients_tensors)):

                gradients_tensor = gradients_tensors[m]
                g_norm = torch.std(gradients_tensor)
                image_tensor.data = image_tensor.data + (lr *(gradients_tensor.data /g_norm) * grad_mask_tensors[m] )## can confirm this is still on the GPU if you have one
        
        else:
           for m in range(len(gradients_tensors)):

                gradients_tensor = gradients_tensors[m]
                g_norm = torch.std(gradients_tensor)
                
                image_tensor.data = image_tensor.data + (lr *(gradients_tensor.data /g_norm) * grad_mask_tensors[m] )## can confirm this is still on the GPU if you have one
        
        image_tensor.data = torch.max(torch.min(image_tensor.data.float(), UPPER_IMAGE_BOUND), LOWER_IMAGE_BOUND).squeeze(0)

    img_out = image_tensor.detach().cpu()

    img_out_np = img_out.numpy()
    img_out_np = img_out_np.transpose(1,2,0)
    
    return img_out_np

def dream_on_octave(model, image_np, layers, iterations, lr,  custom_func = None, max_rotation = 0.2, gradient_smoothing_coeff = None, gradient_smoothing_kernel_size = None, device = None, default_func = None):

    """
    Core function for image optimization with gradient masks 

    input args{
        model: pytorch model 
        image_np: 3 channel numpy image with shape (C,H,W)
        layers: list of layers whose outputs are to be used [model.layer1, model.layer2]
        lr: learning rate
        custom_func: custom loss defined by the user 
        max_rotation: caps the max amount of rotation on the input tensor, helps reduce noise 
        gradient_smoothing_coeff: helps reduce fig frequency noise in gradients before adding
        gradient_smoothing_kernel_size: kernel size to be used while smoothing 
        device: cuda or CPU depending on the availability
        default_func: default func to be used if no custom_func is not given
    }
    """

    image_tensor = pytorch_input_adapter(image_np, device = device)
    for i in range(iterations):
        """
        rolling 
        """

        roll_x, roll_y = find_random_roll_values_for_tensor(image_tensor)

        image_tensor_rolled = roll_torch_tensor(image_tensor, roll_x, roll_y) 
        
        """
        rotating
        """
        
        theta = get_random_rotation_angle(theta_max= max_rotation)


        image_tensor_rolled_rotated = rotate_image_tensor(image_tensor = image_tensor_rolled, theta = theta, device = device)

        """
        getting gradients
        """
        gradients_tensor = get_gradients(net_in = image_tensor_rolled_rotated, net = model, layers = layers, default_func = default_func ,custom_func= custom_func).detach()

        """
        unrotate and unroll gradients of the image tensor
        """
        gradients_tensor_unrotated  = rotate_image_tensor(gradients_tensor, theta = -theta, device = device)
        gradients_tensor = roll_torch_tensor(gradients_tensor_unrotated, -roll_x, -roll_y)  

        """
        image update
        """
        
        if gradient_smoothing_kernel_size is not None and gradient_smoothing_coeff is not None:
            
            sigma = ((i + 1) / iterations) * 2.0 + gradient_smoothing_coeff

            smooth_gradients_tensor = CascadeGaussianSmoothing(kernel_size = gradient_smoothing_kernel_size, sigma = sigma, device = device)(gradients_tensor.unsqueeze(0)).squeeze(0)
            g_norm = torch.std(smooth_gradients_tensor)

            image_tensor.data = image_tensor.data + lr * (smooth_gradients_tensor.data / g_norm) ## can confirm this is still on the GPU if you have one
        else:
            g_norm = torch.std(gradients_tensor)
            image_tensor.data = image_tensor.data + lr *(gradients_tensor.data /g_norm) ## can confirm this is still on the GPU if you have one

        image_tensor.data = torch.max(torch.min(image_tensor.data, UPPER_IMAGE_BOUND), LOWER_IMAGE_BOUND).squeeze(0)
       
    img_out = image_tensor.detach().cpu()

    img_out_np = img_out.numpy()
    img_out_np = img_out_np.transpose(1,2,0)
    
    return img_out_np