import torch
import torch.nn as nn
from torchvision import models
import numpy as np
import matplotlib.pyplot as plt
]import os
import tqdm
from torchvision import transforms
import matplotlib.pyplot as plt
from tqdm import tqdm 
import cv2 

import torch_dreams

mode = "resnet"

image_main = cv2.imread("sample_images/cloudy-mountains.jpg")
image_sample = cv2.cvtColor(image_main, cv2.COLOR_BGR2RGB)
image_sample = cv2.resize(image_sample, (1024,1024))

plt.imshow(image_sample)
plt.show()

if mode == "vgg":
    model= models.vgg19(pretrained=True)
    layers = list(model.features.children())
    model.eval()

    preprocess = torch_dreams.preprocess_func_vgg
    deprocess = torch_dreams.deprocess_func_vgg

else:
    model = models.resnet18(pretrained=True)
    layers = list(model.children())
    model.eval()

    preprocess = torch_dreams.preprocess_func
    deprocess = None


layer = layers[8]

dreamed = torch_dreams.deep_dream(
                        image_np =image_sample, 
                        model = model,
                        layer = layer, 
                        octave_scale = 1.5, 
                        num_octaves = 7, 
                        iterations = 5, 
                        lr = 0.09,
                        preprocess_func = preprocess,
                        deprocess_func = deprocess
                        )

plt.imshow(dreamed)
plt.show()

cv2.imwrite('dream.jpg', dreamed)