import cv2
import numpy as np
import matplotlib.pyplot as plt

from torch_dreams.dreamer import dreamer
import torchvision.models as models

model = models.inception_v3(pretrained=True)
dreamy_boi = dreamer(model)

layer = model.Mixed_6c.branch7x7_1.conv
layer_2 = model.Mixed_6c.branch7x7dbl_3.conv


layers_to_use = [layer, layer_2]

def my_custom_func(layer_outputs):
    
    loss = layer_outputs[0][70].norm()
    return loss

config = {
    "image_path": "images/sample_small.jpg",
    "layers": layers_to_use,
    "octave_scale": 1.1,
    "num_octaves": 11,
    "iterations": 20,
    "lr": 0.03,
    "custom_func": my_custom_func,
    "max_rotation": 0.2,
    "grayscale": False,
    "gradient_smoothing_coeff": 0.5,
    "gradient_smoothing_kernel_size": 9
}

out_single_conv = dreamy_boi.deep_dream(config)

plt.imshow(out_single_conv)
plt.show()