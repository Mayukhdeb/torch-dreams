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
from .image_transforms import transform_to_tensor
from .utils import find_random_roll_values_for_tensor
from .utils import roll_torch_tensor
from .utils import post_process_numpy_image



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

    def default_func(self, layer_outputs):
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
            loss = self.default_func(layer_outputs)


        loss.backward()
        return net_in.grad.data.squeeze()


    def dream_on_octave(self, image_np, layers, iterations, lr, custom_func = None):

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

            roll_x, roll_y = find_random_roll_values_for_tensor(image_tensor)
            image_tensor_rolled = roll_torch_tensor(image_tensor, roll_x, roll_y) 
            gradients_tensor = self.get_gradients(net_in = image_tensor_rolled, net = self.model, layers = layers, custom_func= custom_func).detach()
            gradients_tensor = roll_torch_tensor(gradients_tensor, -roll_x, -roll_y)  
            image_tensor.data = image_tensor.data + lr * gradients_tensor.data ## can confirm this is still on the GPU if you have one

        img_out = image_tensor.detach().cpu()

        img_out_np = img_out.numpy()
        img_out_np = img_out_np.transpose(1,2,0)
        
        return img_out_np


    def deep_dream(self, image_path, layers, octave_scale, num_octaves, iterations, lr, size = None, custom_func = None):

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


        original_image = load_image(image_path)
        original_size = original_image.shape[:-1]
        image_np = preprocess_numpy_img(original_image)

        
        for n in tqdm(range(-num_octaves, 1)):
            
            octave_size = tuple( np.array(original_size) * octave_scale**n)
            new_size = (int(octave_size[1]), int(octave_size[0]))

            image_np = cv2.resize(image_np, new_size)

            image_np = self.dream_on_octave(image_np  = image_np, layers = layers, iterations = iterations, lr = lr, custom_func = custom_func)

        image_np = post_process_numpy_image(image_np)
        return image_np