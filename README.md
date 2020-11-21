# torch-dreams
Making deep neural networks more interpretable, one octave at a time.

:exclamation::exclamation: **warning**: This project is undergoing a major overhaul right now. So most of the backend would be broken. 

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Mayukhdeb/torch-dreams/blob/master/notebooks/torch_dreams_examples.ipynb)
[![](https://img.shields.io/github/last-commit/mayukhdeb/torch-dreams)](https://github.com/mayukhdeb/torch-dreams/commits/master)
```
pip install torch-dreams --upgrade
```

## Contents:

  * **You might want to read**:
    * [Feature visualization by Olah, et al.](https://distill.pub/2017/feature-visualization/)
    * [Google AI blog on DeepDreams](https://ai.googleblog.com/2015/06/inceptionism-going-deeper-into-neural.html)
    
  * **Notebooks**:
    * [optimizing random noise to activate channels within the inceptionv3](https://nbviewer.jupyter.org/github/Mayukhdeb/torch-dreams-notebooks/blob/main/notebooks/torch_dreams_showcase_inceptionv3.ipynb)
    * [optimizing images to activate channels within googlenet](https://nbviewer.jupyter.org/github/Mayukhdeb/torch-dreams-notebooks/blob/main/notebooks/torch_dreams_showcase_googlenet.ipynb)
    
  * **Images**:
    * [Full size images from notebooks](https://github.com/Mayukhdeb/torch-dreams-notebooks/tree/main/images/raw_output)

## Optimizing noise to activate multiple channels simultaneously within the `inceptionv3`

<code><img src = "https://github.com/Mayukhdeb/torch-dreams/blob/master/images/visualizations/channel_blending_inceptionv3.jpg?raw=true" width = "65%"></code>

## Feature visualization through combined optimization of channels 

<code><img src = "https://github.com/Mayukhdeb/torch-dreams/blob/master/images/visualizations/channel_blending_googlenet_2.jpg?raw=true" width = "45%"></code>
<code><img src = "https://github.com/Mayukhdeb/torch-dreams/blob/master/images/visualizations/channel_blending_googlenet.jpg?raw=true" width = "45%"></code>

## Changes under way:
1. Expand`torch_dreams` to facilitate research in neural network interpretability.

