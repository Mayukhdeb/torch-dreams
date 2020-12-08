# Torch-Dreams
Making deep neural networks more interpretable for research, and for art. 

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Mayukhdeb/torch-dreams-notebooks/blob/main/notebooks/torch_dreams_starter.ipynb)
[![](https://img.shields.io/github/last-commit/mayukhdeb/torch-dreams)](https://github.com/mayukhdeb/torch-dreams/commits/master)

<img src = "https://raw.githubusercontent.com/Mayukhdeb/torch-dreams/master/images/inceptionv3_new_channels.jpeg">

```
pip install torch-dreams --upgrade
```

## Quick start
> Make sure you also check out the [quick start colab notebook]() and the [docs]() for more interesting examples. 
```python
import matplotlib.pyplot as plt
import torchvision.models as models
from torch_dreams.dreamer import dreamer

model = models.inception_v3(pretrained=True)
dreamy_boi = dreamer(model)

config = {
    "image_path": "your_image.jpg",
    "layers": [model.Mixed_6c.branch1x1],
    "octave_scale": 1.2,
    "num_octaves": 10,
    "iterations": 20,
    "lr": 0.03,
    "max_rotation": 0.5,
}

out = dreamy_boi.deep_dream(config)
plt.imshow(out)
plt.show()
```

## A closer look

The `config` is where we get to customize how exactly we want the optimization to happen. Here's an example without using gradient masks:

```python
config = {
    "image_path": "your_image.jpg",
    "layers": [model.Mixed_6c.branch1x1],
    "octave_scale": 1.2,
    "num_octaves": 10,
    "iterations": 20,
    "lr": 0.03,
    "custom_func": None,
    "max_rotation": 0.5,
    "gradient_smoothing_coeff": 0.1,
    "gradient_smoothing_kernel_size": 3
}
```

* `image_path`: specifies the relative path to the input image. 

* `layers`: List of layers whose outputs are to be "stored" for optimization later on. For example, if we want to use 2 layers:
    ```python
    config["layers"] = [
    model.Mixed_6d.branch1x1,
    model.Mixed_5c
    ]
    ```
    
* `octave_scale`: Factor by which the image is scaled up after each octave. 
* `num_octaves`: Number of times the image is scaled up in order to reach back to the original size.
* `iterations`: Number of gradient ascent steps taken per octave. 
* `lr`: Learning rate used in each step of the gradient ascent. 
* `custom_func` (optional): Use this to build your own custom optimization functions to optimize on individual channels/units/etc.

    For example, if we want to optimize the th **47th channel of the first layer** and the **22nd channel of the 2nd layer** simultaneously:

    ```python
    
    def my_custom_func(layer_outputs):
        loss = layer_outputs[0][47].mean() + layer_outputs[1][22].mean()
        return loss
    config["custom_func"] = my_custom_func
    ```
* `max_rotation` (optional): Caps the maximum rotation to apply on the image.
* `gradient_smoothing_coeff` (optional): Higher -> stronger blurring. 
* `gradient_smoothing_kernel_size`: (optional) Kernel size to be used when applying gaussian blur.

## Important links:

* [Feature visualization by Olah, et al.](https://distill.pub/2017/feature-visualization/)
* [Google AI blog on DeepDreams](https://ai.googleblog.com/2015/06/inceptionism-going-deeper-into-neural.html)


