# torch-dreams
Making deep neural networks more interpretable, one octave at a time.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Mayukhdeb/torch-dreams-notebooks/blob/main/notebooks/torch_dreams_starter.ipynb)
[![](https://img.shields.io/github/last-commit/mayukhdeb/torch-dreams)](https://github.com/mayukhdeb/torch-dreams/commits/master)
```
pip install torch-dreams --upgrade
```

<img src = "https://github.com/Mayukhdeb/torch-dreams/blob/master/images/inceptionv3_cherrypicked_channels.jpg?raw=true">


## Contents:

  * **You might want to read**:
    * [Feature visualization by Olah, et al.](https://distill.pub/2017/feature-visualization/)
    * [Google AI blog on DeepDreams](https://ai.googleblog.com/2015/06/inceptionism-going-deeper-into-neural.html)
  * **Notebooks**:
    * [Quick start notebook](https://colab.research.google.com/github/Mayukhdeb/torch-dreams-notebooks/blob/main/notebooks/torch_dreams_starter.ipynb)
    * [optimizing random noise to activate channels within the inceptionv3](https://nbviewer.jupyter.org/github/Mayukhdeb/torch-dreams-notebooks/blob/main/notebooks/torch_dreams_showcase_inceptionv3.ipynb)
    * [optimizing images to activate channels within googlenet](https://nbviewer.jupyter.org/github/Mayukhdeb/torch-dreams-notebooks/blob/main/notebooks/torch_dreams_showcase_googlenet.ipynb)
  * **Images**:
    * [Full size images from notebooks](https://github.com/Mayukhdeb/torch-dreams-notebooks/tree/main/images/raw_output)
    
## Quick start
> This is a very simple example. For advanced functionalities like simultaneous optimization of channels/layers/units, check out the [quick start notebook](https://colab.research.google.com/github/Mayukhdeb/torch-dreams-notebooks/blob/main/notebooks/torch_dreams_starter.ipynb)
* Importing the good stuff
```python
import os
import matplotlib.pyplot as plt
import torchvision.models as models 
from torch_dreams.dreamer import dreamer
```
* Initiating `torch_dreams.dreamer` and selecting a layer to optimize on
```python
model = models.inception_v3(pretrained=True)
dreamy_boi = dreamer(model)

layer = model.Mixed_5d
layers_to_use = [layer]  ## feel free to add more layers
```
* Showtime
```python
os.system("wget https://raw.githubusercontent.com/Mayukhdeb/torch-dreams/master/images/noise.jpg")

out_single_layer = dreamy_boi.deep_dream(
    image_path = "noise.jpg",
    layers = layers_to_use,
    octave_scale = 1.3,
    num_octaves = 7,
    iterations = 100,
    lr = 0.9
)

plt.imshow(out_single_layer)
plt.show()
```
## Interpolating though different classes by weighted optimization of random noise 
<img src = "https://github.com/Mayukhdeb/torch-dreams/blob/master/images/resnet18_goldfish_zebra_baloon_interp.jpg?raw=true">


## Optimizing noise to activate multiple channels simultaneously within the `inceptionv3`

<code><img src = "https://github.com/Mayukhdeb/torch-dreams/blob/master/images/visualizations/channel_blending_inceptionv3.jpg?raw=true" width = "45%"></code>
<code><img src = "https://github.com/Mayukhdeb/torch-dreams/blob/master/images/visualizations/channel_blending_inceptionv3_2.jpg?raw=true" width = "45%"></code>

## Feature visualization through combined optimization of channels 

<code><img src = "https://github.com/Mayukhdeb/torch-dreams/blob/master/images/visualizations/channel_blending_googlenet_2.jpg?raw=true" width = "45%"></code>
<code><img src = "https://github.com/Mayukhdeb/torch-dreams/blob/master/images/visualizations/channel_blending_googlenet.jpg?raw=true" width = "45%"></code>

## Changes under way:
1. Expand`torch_dreams` to facilitate research in neural network interpretability.

