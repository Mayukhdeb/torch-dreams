from torchvision import models
import matplotlib.pyplot as plt
import cv2
import torch

from torch_dreams import  utils
from torch_dreams import dreamer

model= models.vgg19(pretrained=True)
layers = list(model.features.children())

layers_to_use= layers[25:28]

dreamy_boi = dreamer(model = model)

def my_custom_func(layer_outputs):
    """
    this custom func would get applied to the list of layer outputs

    the layers whose outputs are given here are the ones you asked for in the layers arg
    """
    # print([l.size() for l in layer_outputs])
    loss = layer_outputs[1][100].norm()
    return loss

out = dreamy_boi.deep_dream(
    image_path = "sample_small.jpg",
    layers = layers_to_use,
    octave_scale = 1.4,
    num_octaves = 3,
    iterations =10,
    lr = 0.85,
    custom_func = my_custom_func
)

plt.imshow(out)
plt.show()

out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)

cv2.imwrite("dream.jpg", out*255)

