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

from .constants import UPPER_IMAGE_BOUND_GRAY
from .constants import LOWER_IMAGE_BOUND_GRAY

from .constants import default_config

from .dreamer_utils import default_func_norm
from .dreamer_utils import get_gradients


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
        self.config = default_config
        self.default_func = default_func_norm
        self.get_gradients  = get_gradients

        print("dreamer init on: ", self.device)

    def dream_on_octave(self, image_np, layers, iterations, lr, custom_func = None, max_rotation = 0.2, gradient_smoothing_coeff = None, gradient_smoothing_kernel_size = None, grayscale = False):

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
            gradients_tensor = self.get_gradients(net_in = image_tensor_rolled_rotated, net = self.model, layers = layers, default_func = self.default_func ,custom_func= custom_func).detach()

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

                smooth_gradients_tensor = CascadeGaussianSmoothing(kernel_size = gradient_smoothing_kernel_size, sigma = sigma, device = self.device, grayscale= grayscale)(gradients_tensor.unsqueeze(0)).squeeze(0)
                g_norm = torch.std(smooth_gradients_tensor)

                image_tensor.data = image_tensor.data + lr * (smooth_gradients_tensor.data / g_norm) ## can confirm this is still on the GPU if you have one
            else:
                g_norm = torch.std(gradients_tensor)
                image_tensor.data = image_tensor.data + lr *(gradients_tensor.data /g_norm) ## can confirm this is still on the GPU if you have one

            if grayscale is False:

                image_tensor.data = torch.max(torch.min(image_tensor.data, UPPER_IMAGE_BOUND), LOWER_IMAGE_BOUND).squeeze(0)
            else:
                image_tensor.data = torch.max(torch.min(image_tensor.data, UPPER_IMAGE_BOUND_GRAY), LOWER_IMAGE_BOUND_GRAY).squeeze(0)


        img_out = image_tensor.detach().cpu()

        img_out_np = img_out.numpy()
        img_out_np = img_out_np.transpose(1,2,0)
        
        return img_out_np


    def dream_on_octave_with_masks(self, image_np, layers, iterations, lr, custom_funcs = [None], max_rotation = 0.2, gradient_smoothing_coeff = None, gradient_smoothing_kernel_size = None, grad_mask =None, grayscale = False):

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
            grad_mask_tensors = [pytorch_input_adapter(g_mask, device = self.device).double() for g_mask in grad_mask]


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

            gradients_tensors = []
            for c in range(len(custom_funcs)):

                gradients_tensor = self.get_gradients(net_in = image_tensor_rolled_rotated, net = self.model, layers = layers,default_func = self.default_func ,custom_func= custom_funcs[c]).detach()
                gradients_tensors.append(gradients_tensor)
            """
            unrotate and unroll gradients of the image tensor
            """
            gradients_tensors_unrotated  = [rotate_image_tensor(g, theta = -theta, device = self.device) for g in gradients_tensors]
            gradients_tensors = [roll_torch_tensor(g, -roll_x, -roll_y)  for g in gradients_tensors_unrotated]

            """
            image update
            """
            # print(gradient_smoothing_sigma, gradient_smoothing_kernel_size)
            
            if gradient_smoothing_kernel_size is not None and gradient_smoothing_coeff is not None:
                
                sigma = ((i + 1) / iterations) * 2.0 + gradient_smoothing_coeff

                gradients_tensors = [CascadeGaussianSmoothing(kernel_size = gradient_smoothing_kernel_size, sigma = sigma, device = self.device, grayscale= grayscale)(gradients_tensor.unsqueeze(0)).squeeze(0) for gradients_tensor in gradients_tensors]


                for m in range(len(gradients_tensors)):

                    gradients_tensor = gradients_tensors[m]
                    g_norm = torch.std(gradients_tensor)

                    
                    image_tensor.data = image_tensor.data + (lr *(gradients_tensor.data /g_norm) * grad_mask_tensors[m] )## can confirm this is still on the GPU if you have one
            
            else:
                image_tensor.data = image_tensor.data + lr *(gradients_tensor.data /g_norm) ## can confirm this is still on the GPU if you have one

            image_tensor.data = torch.max(torch.min(image_tensor.data.float(), UPPER_IMAGE_BOUND), LOWER_IMAGE_BOUND).squeeze(0)

        img_out = image_tensor.detach().cpu()

        img_out_np = img_out.numpy()
        img_out_np = img_out_np.transpose(1,2,0)
        
        return img_out_np

    def deep_dream(self, config):


        for key in list(config.keys()):
            self.config[key] = config[key]

        image_path = self.config["image_path"]
        layers =  self.config["layers"]
        octave_scale = self.config["octave_scale"]
        num_octaves = self.config["num_octaves"]
        iterations = self.config["iterations"]
        lr = self.config["lr"]
        custom_func = self.config["custom_func"]
        max_rotation = self.config["max_rotation"]
        grayscale = self.config["grayscale"]
        gradient_smoothing_coeff = self.config["gradient_smoothing_coeff"]
        gradient_smoothing_kernel_size = self.config["gradient_smoothing_kernel_size"]



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
            image_np = self.dream_on_octave(image_np  = image_np, layers = layers, iterations = iterations, lr = lr, custom_func = custom_func, max_rotation= max_rotation, gradient_smoothing_coeff= gradient_smoothing_coeff, gradient_smoothing_kernel_size=gradient_smoothing_kernel_size, grayscale=grayscale)

        image_np = post_process_numpy_image(image_np)
        return image_np

    def deep_dream_with_masks(self, config):


        for key in list(config.keys()):
                    self.config[key] = config[key]

        image_path = self.config["image_path"]
        layers =  self.config["layers"]
        octave_scale = self.config["octave_scale"]
        num_octaves = self.config["num_octaves"]
        iterations = self.config["iterations"]
        lr = self.config["lr"]
        custom_funcs = self.config["custom_func"]
        max_rotation = self.config["max_rotation"]
        grayscale = self.config["grayscale"]
        gradient_smoothing_coeff = self.config["gradient_smoothing_coeff"]
        gradient_smoothing_kernel_size = self.config["gradient_smoothing_kernel_size"]
        grad_mask = self.config["grad_mask"]

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
                grad_mask = [cv2.resize(g, new_size) for g in grad_mask]
            image_np = self.dream_on_octave_with_masks(image_np  = image_np, layers = layers, iterations = iterations, lr = lr, custom_funcs = custom_funcs, max_rotation= max_rotation, gradient_smoothing_coeff= gradient_smoothing_coeff, gradient_smoothing_kernel_size=gradient_smoothing_kernel_size, grad_mask= grad_mask, grayscale=grayscale )

        image_np = post_process_numpy_image(image_np)
        return image_np