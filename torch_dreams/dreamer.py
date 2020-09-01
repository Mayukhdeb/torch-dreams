import torch
import torch.nn as nn
from torchvision import models
import numpy as np
import matplotlib.pyplot as plt
import os
import tqdm
from torchvision import transforms
import matplotlib.pyplot as plt
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

    def __init__(self, model, preprocess_func, deprocess_func = None):
        self.model = model
        self.model = self.model.eval()
        self.preprocess_func = preprocess_func
        self.deprocess_func = deprocess_func
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    
    def get_gradients(self, net_in, net, layer, out_channels = None):     
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


    def dream_on_octave(self, image_tensor, layer, iterations, lr, out_channels = None):

        image_tensor = self.preprocess_func(image_tensor).to(self.device)

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
        original_size = image_np.shape[:2]

        for n in tqdm(range(-num_octaves, 1)):
            
            octave_size = tuple( np.array(original_size) * octave_scale**n)
            new_size = (int(octave_size[1]), int(octave_size[0]))
            image_np = cv2.resize(image_np, new_size)
            image_np = self.dream_on_octave(image_np, layer =  layer, iterations = iterations, lr = lr, out_channels = None)

        image_np_normalised = cv2.normalize(image_np, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F).astype(np.uint8)

        return image_np_normalised

    def deep_dream_on_video(self, video_path, save_name , layer, octave_scale, num_octaves, iterations, lr, size = None,  framerate = 30 ):

        all_frames = video_to_np_arrays(video_path, skip_value = 1, size = None)[:4] ## [:5] is for debugging
        all_dreams = []

        for i in range(len(all_frames)):
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