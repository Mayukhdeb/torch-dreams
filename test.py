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

from torch_dreams import  utils
from torch_dreams import dreamer
from torch_dreams.simple import vgg19_dreamer



mode = "vgg"

image_main = cv2.imread("torch_dreams/sample_images/cloudy-mountains.jpg")
image_sample = cv2.cvtColor(image_main, cv2.COLOR_BGR2RGB)
image_sample = cv2.resize(image_sample, (1024,1024))

plt.imshow(image_sample)
plt.show()

if mode == "vgg":
    model= models.vgg19(pretrained=True)
    layers = list(model.features.children())
    model.eval()

    preprocess = utils.preprocess_func_vgg
    deprocess = utils.deprocess_func_vgg
    layer = layers[34]


else:
    model = models.resnet18(pretrained=True)
    layers = list(model.children())
    model.eval()

    preprocess = utils.preprocess_func
    deprocess = None

    layer = layers[8]

dreamer = dreamer(model, preprocess, deprocess)


dreamed = dreamer.deep_dream(
                        image_np =image_sample, 
                        layer = layer, 
                        octave_scale = 1.5, 
                        num_octaves = 2, 
                        iterations = 2, 
                        lr = 0.09,
                        )

plt.imshow(dreamed)
plt.show()

cv2.imwrite('dream.jpg', dreamed)


"""
Simple dreamer
"""
simple_dreamer = vgg19_dreamer()

dreamed_image = simple_dreamer.dream(
    image_np = image_sample,
    layer_index= 27,
    iterations= 3
)

plt.imshow(dreamed_image)
plt.show()

