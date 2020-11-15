import torch
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

model= models.vgg19(pretrained=True)
layers = list(model.features.children())

layer = layers[24]

dreamy_boi = dreamer(model = model)

out = dreamy_boi.deep_dream(
    image_path = "sample.jpg",
    layer = layer,
    octave_scale = 1.3,
    num_octaves = 7,
    iterations =15,
    lr = 0.15
)

plt.imshow(out)
plt.show()

out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)

cv2.imwrite("dream.jpg", out*255)

