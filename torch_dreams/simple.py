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
from .dreamer import *

class vgg19_dreamer(dreamer):

    def __init__(self):
        super().__init__(
            model = models.vgg19(pretrained=True), 
            preprocess_func =  preprocess_func,   ## for some odd reason, preprocess_func works and preprocess_func_vgg does not
            deprocess_func = None
            )
        self.layers = list(self.model.features.children())
        
    def show_layers(self):
        print(self.model )

    def dream(self, image_path, layer_index = 13, octave_scale = 1.4, num_octaves = 2, iterations = 30, lr= 0.09, size = None):

        image_np = cv2.imread(image_path)

        if size is not None:
            image_np = cv2.resize(image_np, size) 

      
        dream_normalised = self.deep_dream(
                                        image_np = image_np,
                                        layer = self.layers[layer_index],
                                        octave_scale = octave_scale,
                                        num_octaves = num_octaves,
                                        iterations = iterations,
                                        lr = lr
               
                        )
   
        return dream_normalised

    def progressive_dream(self, image_path, save_name, layer_index = 13, octave_scale = 1.4, num_octaves = 2, iterations = 30, lower_lr = 0.0, upper_lr = 0.09, num_steps = 15, framerate = 15, size = None):


        self.progressive_deep_dream(
            image_path = image_path,
            save_name = save_name,
            layer = self.layers[layer_index],
            octave_scale = octave_scale,
            num_octaves = num_octaves,
            iterations = iterations,
            lower_lr = lower_lr,
            upper_lr = upper_lr,
            num_steps = num_steps,
            framerate = framerate,
            size = size
        )

