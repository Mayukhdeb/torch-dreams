import torch
from torchvision import models
import numpy as np
import os
import tqdm
from torchvision import transforms
from tqdm import tqdm 
import cv2 

from  .utils import *


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

    def __init__(self, model, preprocess_func, deprocess_func = None):
        self.model = model
        self.model = self.model.eval()
        self.preprocess_func = preprocess_func
        self.deprocess_func = deprocess_func
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device) ## model moves to GPU if available

    
    def get_gradients(self, net_in, net, layer, out_channels = None):   
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
        hook = Hook(layer)
        net_out = net(net_in)
        if out_channels == None:
            loss = hook.output[0].norm()
        else:
            loss = hook.output[0][out_channels].norm()
        loss.backward()
        return net_in.grad.data.squeeze()


    def dream_on_octave(self, image_np, layer, iterations, lr, out_channels = None):

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
        }
        """

        image_tensor = self.preprocess_func(image_np).to(self.device) # image tensor moves to GPU if available

        for i in range(iterations):

            roll_x, roll_y = find_random_roll_values_for_tensor(image_tensor)
            image_tensor_rolled = roll_torch_tensor(image_tensor, roll_x, roll_y) 
            gradients_tensor = self.get_gradients(image_tensor_rolled, self.model, layer, out_channels).detach()
            gradients_tensor = roll_torch_tensor(gradients_tensor, -roll_x, -roll_y)  
            image_tensor.data = image_tensor.data + lr * gradients_tensor.data ## can confirm this is still on the GPU if you have one

        img_out = image_tensor.detach().cpu()

        if self.deprocess_func is not None:
            img_out = deprocess_func(img_out)

        img_out_np = img_out.numpy()
        img_out_np = img_out_np.transpose(1,2,0)
        
        return img_out_np


    def deep_dream(self, image_np, layer, octave_scale, num_octaves, iterations, lr):

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

        original_size = image_np.shape[:2]

        for n in range(-num_octaves, 1):
            
            octave_size = tuple( np.array(original_size) * octave_scale**n)
            new_size = (int(octave_size[1]), int(octave_size[0]))

            image_np = cv2.resize(image_np, new_size)
            image_np = self.dream_on_octave(image_np, layer =  layer, iterations = iterations, lr = lr, out_channels = None)
            
                    
        image_np = cv2.convertScaleAbs(image_np, alpha = 255)

        
        return image_np

    def deep_dream_on_video(self, video_path, save_name , layer, octave_scale, num_octaves, iterations, lr, size = None,  framerate = 30, skip_value = 1 ):

        """
        Runs deep-dreams on each frame of a video and returns another video made of the dee-dream frames

        input args{
            video_path = "path/to/video.mp4"
            save_name = filename of the deep-dream video file that would be saved 
            layer = specifies the layer whose activations are to be maximized
            octave_scale <ideal range = [1.1, 1.5]> = factor by which the size of an octave image is increased after each octave 
            num_octaves <ideal range = [3, 8]> = number of times the original zero octave image has to expand in order to reack back to it's original size 
            iterations = number of time the original image is added by the gradients multiplied by a constant factor lr
            lr = equivalent to learning rate
            size = specifies the size of each frame for the output video. resizing is done before running the deep-dreams, smaller size is  faster 
            framerate = FPS of the video to be saved 
            skip_value <optional> = set this to n if you want to skip n frames after each frame from the original video, equivalent to skip value in for loops 
        }

        returns{
            Saves a video with filename save_name
        }
        """

        all_frames = video_to_np_arrays(video_path, skip_value = skip_value, size = None)  ## [:5] is for debugging
        all_dreams = []

        for i in tqdm(range(len(all_frames)), desc = "Running deep-dreams video frames: "):
            dreamed = self.deep_dream(
                                    image_np = all_frames[i],
                                    layer = layer,
                                    octave_scale = octave_scale,
                                    num_octaves = num_octaves,
                                    iterations = iterations,
                                    lr = lr
                                )
            all_dreams.append(dreamed)

        
        all_dreams = np.array(all_dreams)
        if size is None:
            size = (all_dreams[0].shape[-2], all_dreams[0].shape[-3]) ## (width, height)
        write_video_from_image_list(save_name = save_name, all_images_np=  all_dreams,framerate = framerate, size = size)




    def progressive_deep_dream(self, image_path, save_name , layer, octave_scale, num_octaves, iterations, lower_lr, upper_lr, num_steps, framerate = 15, size = None):

        """
        Runs deep-dreams on a single image with gradually increasing lr, returns a video of the progressive frames

        input args{
            image_path = "path/to/image.jpg"
            save_name = filename of the video to be saved 
            octave_scale <ideal range = [1.1, 1.5]> = factor by which the size of an octave image is increased after each octave 
            num_octaves <ideal range = [3, 8]> = number of times the original zero octave image has to expand in order to reack back to it's original size 
            iterations = number of time the original image is added by the gradients multiplied by a constant factor lr
            lower_lr = lower limit of learning rate
            upper_lr = upper limit of learning rate
            num_steps = number of steps to be taken between lower and upper lr
            framerate = FPS of the video to be saved 
            size <optional> = (width, height) can be used to resize the original image to a smaller size 
        }

        returns{
            Saves a video with filename save_name
        }
        """

        lrs = np.linspace(lower_lr, upper_lr, num_steps)
        dreams = []

        image_np = cv2.imread(image_path)

        if size is not None:
            image_np = cv2.resize(image_np, size)

        for lr in tqdm(lrs, desc = "Running progressive deep-dream on image: "):
            dreamed_image = self.deep_dream(
                image_np = image_np,
                layer = layer,
                octave_scale = octave_scale,
                num_octaves = num_octaves,
                iterations = iterations,
                lr = lr
            )
            dreams.append(dreamed_image)
        
        dreams = np.array(dreams)

        if size is None:
            size = (dreams[0].shape[-2], dreams[0].shape[-3]) ## (width, height)
        write_video_from_image_list(save_name = save_name, all_images_np=  dreams,framerate = framerate, size = size)
