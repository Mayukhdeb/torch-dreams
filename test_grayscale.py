import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torch_dreams.dreamer import dreamer
import torchvision.models as models

model = models.resnet18(pretrained=False)
model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
model.fc = nn.Linear(in_features=512, out_features=10, bias=True)

model.load_state_dict(torch.load("models/resnet18_mnist.pth"))
dreamy_boi = dreamer(model)

layers_to_use = [
                 model.layer4[0].conv2
            ]

def my_custom_func(layer_outputs):
    # loss = layer_outputs[0][6]   ## 7th label on FC
    loss = layer_outputs[0].norm()  

    return loss


config = {
    "image_path": "images/sample_small.jpg",
    "layers": layers_to_use,
    "octave_scale": 1.69,
    "num_octaves": 7,
    "iterations": 20,
    "lr": 0.09,
    "custom_func": my_custom_func,
    "max_rotation": 0.2,
    "grayscale": True,
    "gradient_smoothing_coeff": 0.5,
    "gradient_smoothing_kernel_size": 9
}

out_single_layer = dreamy_boi.deep_dream(config)

plt.imshow(out_single_layer)
plt.show()