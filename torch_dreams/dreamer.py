import torch
from torchvision import models
import numpy as np
import os
import tqdm
from torchvision import transforms
from tqdm import tqdm 
import cv2 


from  .utils import load_image
from .utils import pytorch_input_adapter
from .utils import preprocess_numpy_img
from .utils import pytorch_output_adapter
from .utils import find_random_roll_values_for_tensor
from .utils import roll_torch_tensor
from .utils import post_process_numpy_image
from .image_transforms import transform_to_tensor

from .utils import get_random_rotation_angle
from .utils import rotate_image_tensor
from .utils import CascadeGaussianSmoothing

from .constants import UPPER_IMAGE_BOUND
from .constants import LOWER_IMAGE_BOUND

class Hook():
    def __init__(self, module, backward=False):
        if backward==False:
            self.hook = module.register_forward_hook(self.hook_fn)
        else:
            self.hook = module.register_backward_hook(self.hook_fn)
    def hook_fn(self, module, input, output):
        self.input = input
        self.output = output
    def close(self):
        self.hook.remove()

class dreamer(object):

    """
    Main class definition for torch-dreams:

    model = Any PyTorch deep-learning model
    preprocess_func = Set of torch transforms required for the model wrapped into a function. See torch_dreams.utils for examples
    deprocess_func <optional> = set of reverse transforms, to be applied before converting the image back to numpy
    device = checks for a GPU, uses the GPU for tensor operations if available
    """

    def __init__(self, model):
        self.model = model
        self.model = self.model.eval()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device) ## model moves to GPU if available

        print("dreamer init on: ", self.device)

    def default_func_MSE(self, layer_outputs):
        losses = []
        for output in layer_outputs:

            loss_component = torch.nn.MSELoss(reduction='mean')(output, torch.zeros_like(output))
            losses.append(loss_component)

        loss = torch.mean(torch.stack(losses))
        return loss
        
    def default_func_norm(self, layer_outputs):
        losses = []
        for output in layer_outputs:
            losses.append(output.norm())
        loss = torch.mean(torch.stack(losses))
        return loss

    def get_gradients(self, net_in, net, layers, custom_func = None):   
        """
        Executes the forward pass through the model and returns the gradients from the selected layer. 

        input args{
            net_in = the 3D tensor which is to be used in the forward pass <size = (C, H, W)>
            net = pytorch model which is being used for the  deep-dream
            layer = layer instance of net whose activations are to be maximized
            out_channels <optional> = manual selection of output channel from the layer that has been chosen
        }

        returns{
            gradient of model weights 
        }
        """  
        net_in = net_in.unsqueeze(0)
        net_in.requires_grad = True
        net.zero_grad()

        hooks = []
        for layer in layers:

            hook = Hook(layer)
            hooks.append(hook)

        net_out = net(net_in)

        layer_outputs = []

        for hook in hooks:

            out = hook.output[0]
            layer_outputs.append(out)

        if custom_func is not None:
            loss = custom_func(layer_outputs)
        else:
            loss = self.default_func_norm(layer_outputs)


        loss.backward()
        return net_in.grad.data.squeeze(0)


    def dream_on_octave(self, image_np, layers, iterations, lr, custom_func = None, max_rotation = 0.2, gradient_smoothing_coeff = None, gradient_smoothing_kernel_size = None, grad_mask =None):

        """
        Deep-dream core function, runs n iterations on a single octave(image)

        input args{
            image_np = 3D numpy array of the image <size = (W, H, C)>
            layer = specifies the layer whose activations are to be maximized
            iterations = number of time the original image is added by the gradients multiplied by a constant factor lr
            out_channels <optinal> = manual selection of output channel from the layer that has been chosen
        }

        returns{
            3D np.array which is basicallly the resulting image after running through one single octave
        }            print(roll_x, roll_y)

        """

        image_tensor = pytorch_input_adapter(image_np, device = self.device)
        if grad_mask is not None:
            grad_mask_tensor = pytorch_input_adapter(grad_mask, device = self.device).double()

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
            image_tensor_rolled_rotated = rotate_image_tensor(image_tensor = image_tensor_rolled, theta = theta, device = self.device)

            """
            getting gradients
            """
            gradients_tensor = self.get_gradients(net_in = image_tensor_rolled_rotated, net = self.model, layers = layers, custom_func= custom_func).detach()

            """
            unrotate and unroll gradients of the image tensor
            """
            gradients_tensor_unrotated  = rotate_image_tensor(gradients_tensor, theta = -theta, device = self.device)
            gradients_tensor = roll_torch_tensor(gradients_tensor_unrotated, -roll_x, -roll_y)  

            """
            image update
            """
            # print(gradient_smoothing_sigma, gradient_smoothing_kernel_size)
            
            if gradient_smoothing_kernel_size is not None and gradient_smoothing_coeff is not None:
                
                sigma = ((i + 1) / iterations) * 2.0 + gradient_smoothing_coeff
                gradients_tensor = CascadeGaussianSmoothing(kernel_size = gradient_smoothing_kernel_size, sigma = sigma, device = self.device)(gradients_tensor.unsqueeze(0)).squeeze(0)
                
            g_norm = torch.std(gradients_tensor)

            if grad_mask is not None:
                
                image_tensor.data = image_tensor.data + (lr *(gradients_tensor.data /g_norm) * grad_mask_tensor )## can confirm this is still on the GPU if you have one
            
            else:
                image_tensor.data = image_tensor.data + lr *(gradients_tensor.data /g_norm) ## can confirm this is still on the GPU if you have one

            image_tensor.data = torch.max(torch.min(image_tensor.data.float(), UPPER_IMAGE_BOUND), LOWER_IMAGE_BOUND).squeeze(0)

        img_out = image_tensor.detach().cpu()

        img_out_np = img_out.numpy()
        img_out_np = img_out_np.transpose(1,2,0)
        
        return img_out_np


    def deep_dream(self, image_path, layers, octave_scale, num_octaves, iterations, lr, size = None, custom_func = None, max_rotation = 0.2, grayscale = False, gradient_smoothing_coeff = 0.5, gradient_smoothing_kernel_size = 5, grad_mask = None):

        """
        High level function used to call the core deep-dream functions on a single image for n octaves.

        input args{
            image_np = 3D numpy array which is basically the input image 
            layer = specifies the layer whose activations are to be maximized
            octave_scale <ideal range = [1.1, 1.5]> = factor by which the size of an octave image is increased after each octave 
            num_octaves <ideal range = [3, 8]> = number of times the original zero octave image has to expand in order to reack back to it's original size 
            iterations = number of time the original image is added by the gradients multiplied by a constant factor lr
            lr = equivalent to learning rate
        } 
        """

        original_image = load_image(image_path, grayscale=grayscale)
        image_np = preprocess_numpy_img(original_image, grayscale=grayscale)
        if grayscale is True:
            image_np = np.expand_dims(image_np, axis = -1)

        original_size = image_np.shape[:-1]

        for n in tqdm(range(-num_octaves, 1)):
            
            octave_size = tuple( np.array(original_size) * octave_scale**n)
            new_size = (int(octave_size[1]), int(octave_size[0]))

            image_np = cv2.resize(image_np, new_size)
            if grayscale is True:
                image_np = np.expand_dims(image_np, axis = -1)
            
            if grad_mask is not None:
                grad_mask = cv2.resize(grad_mask, new_size)
            image_np = self.dream_on_octave(image_np  = image_np, layers = layers, iterations = iterations, lr = lr, custom_func = custom_func, max_rotation= max_rotation, gradient_smoothing_coeff= gradient_smoothing_coeff, gradient_smoothing_kernel_size=gradient_smoothing_kernel_size, grad_mask= grad_mask )

        image_np = post_process_numpy_image(image_np)
        return image_np