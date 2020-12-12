# Torch-Dreams
Making neural networks more interpretable, for research and art. 


[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Mayukhdeb/torch-dreams-notebooks/blob/main/docs_notebooks/hello_torch_dreams.ipynb)
[![](https://img.shields.io/github/last-commit/mayukhdeb/torch-dreams)](https://github.com/mayukhdeb/torch-dreams/commits/master)
[![](https://img.shields.io/twitter/url?label=Docs&style=flat-square&url=https%3A%2F%2Fapp.gitbook.com%2F%40mayukh09%2Fs%2Ftorch-dreams%2F)](https://app.gitbook.com/@mayukh09/s/torch-dreams/)

<img src = "https://github.com/Mayukhdeb/torch-dreams/blob/master/images/banner_1.png?raw=true">

```
pip install torch-dreams 
```

## Contents

* Docs 
    1. [Visualizing individual channels](https://app.gitbook.com/@mayukh09/s/torch-dreams/visualizing-individual-channels)
    2. [Channel Algebra](https://app.gitbook.com/@mayukh09/s/torch-dreams/blending)
    3. [Gradient Masks](https://app.gitbook.com/@mayukh09/s/torch-dreams/gradient-masks)
* Notebooks
    1. [Quick start on colab](https://colab.research.google.com/github/Mayukhdeb/torch-dreams-notebooks/blob/main/docs_notebooks/hello_torch_dreams.ipynb)


## Quick start
> Make sure you also check out the [quick start colab notebook](https://colab.research.google.com/github/Mayukhdeb/torch-dreams-notebooks/blob/main/docs_notebooks/hello_torch_dreams.ipynb) and the [docs](https://app.gitbook.com/@mayukh09/s/torch-dreams/) for more interesting examples. 
```python
import matplotlib.pyplot as plt
import torchvision.models as models
from torch_dreams.dreamer import dreamer

model = models.inception_v3(pretrained=True)
dreamy_boi = dreamer(model)

config = {
    "image_path": "your_image.jpg",
    "layers": [model.Mixed_5c.branch3x3dbl_3],
    "octave_scale": 1.1,
    "num_octaves": 14,
    "iterations": 70,
    "lr": 0.03,
    "max_rotation": 0.5,
}

out = dreamy_boi.deep_dream(config)
plt.imshow(out)
plt.show()
```
---

## Visualizing individual channels

This section of torch_dreams was highly inspired by [Feature visualization by Olah, et al](https://distill.pub/2017/feature-visualization/). We basically optimize the input image to maximize activations of a certain channel of a layer in the neural network. 

First, let's select the layer(s) we want to work on. Feel free to play around with other layers. 

```python
layers_to_use = [model.Mixed_6c.branch7x7_1.conv]
```

The next step now would be to define a `custom_func` that would enable use to selectively optimize a single channel. 


```python 
def my_custom_func(layer_outputs):
    loss = layer_outputs[0][7].mean()  ## 7th channel of first layer from layers_to_use
    return loss
```

The rest is actually very similar to the quick start snippet:

```python
config = {
    "image_path": "noise.jpg",
    "layers": layers_to_use,
    "octave_scale": 1.1,  
    "num_octaves": 20,  
    "iterations": 100,  
    "lr": 0.04,
    "max_rotation": 0.7,
    "custom_func":  my_custom_func,
}

out = dreamy_boi.deep_dream(config)
plt.imshow(out)
plt.show()
```
If things go as planned, you will end up with something like:

<img src = "https://raw.githubusercontent.com/Mayukhdeb/torch-dreams-notebooks/main/images/raw_output/inceptionv3_channels/inceptionv3.Mixed_6c.branch7x7_1.conv_channel_7.jpg" width = "30%">

---
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


