# torch-dreams
Generate deep-dreams with images and videos, best served with CUDA

:exclamation::exclamation: **warning**: This project is undergoing a major overhaul right now. So most of the backend would be broken. 

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Mayukhdeb/torch-dreams/blob/master/notebooks/torch_dreams_examples.ipynb)
[![](https://img.shields.io/github/last-commit/mayukhdeb/torch-dreams)](https://github.com/mayukhdeb/torch-dreams/commits/master)
```
pip install torch-dreams --upgrade
```

<img src = "https://raw.githubusercontent.com/Mayukhdeb/torch-dreams/master/images/tiger_dream.png">
<!--
<code><img src = "https://raw.githubusercontent.com/Mayukhdeb/torch-dreams/master/images/island_deep_dream.gif" width = "45%"></code>
<code><img src = "https://raw.githubusercontent.com/Mayukhdeb/torch-dreams/master/images/tiger_cover.gif" width = "45%"></code>
-->

## Changes under way:
1. Remove `preprocess` and make `torch_dreams.dreamer` directly compatible with `torch.tensor`
2. Expand to `torch_dreams.lucid` to facilitate research in neural network interpretability.

## Less lines of code, more deep-dreams

```python
from torch_dreams.simple import vgg19_dreamer
import cv2 ## for saving images

simple_dreamer = vgg19_dreamer()

dreamed_image = simple_dreamer.dream(
    image_path = "your_image.jpg",
    layer_index= 13,
    iterations= 5,
    size = (256,256)
)

cv2.imwrite("dream.jpg", dreamed_image)
```


## Features:
* Easy to use with `torch_dreams.simple`
* Works on the GPU
* No need to spend hours writing/debugging boilerplate code while slowly forgetting what social a life is.

## Stuff to be added:
* progressive deep-dreams
* optionally trimming video at certain time values for shorter deep-dream videos
