# Torch-Dreams
Making neural networks more interpretable, for research and art. 


[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Mayukhdeb/torch-dreams-notebooks/blob/main/docs_notebooks/hello_torch_dreams.ipynb)
[![build](https://github.com/Mayukhdeb/torch-dreams/actions/workflows/main.yml/badge.svg)](https://github.com/Mayukhdeb/torch-dreams/actions/workflows/main.yml)
[![codecov](https://codecov.io/gh/Mayukhdeb/torch-dreams/branch/master/graph/badge.svg?token=krU6dNleoJ)](https://codecov.io/gh/Mayukhdeb/torch-dreams)
[![Downloads](https://pepy.tech/badge/torch-dreams/month)](https://pepy.tech/project/torch-dreams)
<!-- [![](https://img.shields.io/twitter/url?label=Docs&style=flat-square&url=https%3A%2F%2Fapp.gitbook.com%2F%40mayukh09%2Fs%2Ftorch-dreams%2F)](https://app.gitbook.com/@mayukh09/s/torch-dreams/) -->


<img src = "https://github.com/Mayukhdeb/torch-dreams/blob/master/images/banner_segmentation_model.png?raw=true">

```
pip install torch-dreams 
```

## Contents:

* [Minimal example](https://github.com/Mayukhdeb/torch-dreams#minimal-example)
* [Not so minimal example](https://github.com/Mayukhdeb/torch-dreams#not-so-minimal-example)
* [Visualizing individual channels with `custom_func`](https://github.com/Mayukhdeb/torch-dreams#visualizing-individual-channels-with-custom_func)
* [Visualize features from multiple models simultaneously](https://github.com/Mayukhdeb/torch-dreams#visualize-features-from-multiple-models-simultaneously)
* [Use custom transforms](https://github.com/Mayukhdeb/torch-dreams#using-custom-transforms)
* [Feedback loops](https://github.com/Mayukhdeb/torch-dreams#you-can-also-use-outputs-of-one-render-as-the-input-of-another-to-create-feedback-loops)
* [Other conveniences](https://github.com/Mayukhdeb/torch-dreams#other-conveniences)

## Minimal example
> Make sure you also check out the [quick start colab notebook](https://colab.research.google.com/github/Mayukhdeb/torch-dreams-notebooks/blob/main/docs_notebooks/hello_torch_dreams.ipynb) 


```python
import matplotlib.pyplot as plt
import torchvision.models as models
from torch_dreams.dreamer import dreamer

model = models.inception_v3(pretrained=True)
dreamy_boi = dreamer(model, device = 'cuda')

image_param = dreamy_boi.render(
    layers = [model.Mixed_5b],
)

plt.imshow(image_param)
plt.show()
```

## Not so minimal example
```python
model = models.inception_v3(pretrained=True)
dreamy_boi = dreamer(model, device = 'cuda', quiet = False)

image_param = dreamy_boi.render(
    layers = [model.Mixed_5b],
    width = 256,
    height = 256,
    iters = 150,
    lr = 9e-3,
    rotate_degrees = 15,
    scale_max = 1.2,
    scale_min =  0.5,
    translate_x = 0.2,
    translate_y = 0.2,
    custom_func = None,
    weight_decay = 1e-2,
    grad_clip = 1.,
)

plt.imshow(image_param)
plt.show()
```

## Visualizing individual channels with `custom_func`

```python
model = models.inception_v3(pretrained=True)
dreamy_boi = dreamer(model, device = 'cuda')

layers_to_use = [model.Mixed_6b.branch1x1.conv]

def make_custom_func(layer_number = 0, channel_number= 0): 
    def custom_func(layer_outputs):
        loss = layer_outputs[layer_number][channel_number].mean()
        return -loss
    return custom_func

my_custom_func = make_custom_func(layer_number= 0, channel_number = 119)

image_param = dreamy_boi.render(
    layers = layers_to_use,
    custom_func = my_custom_func,
)
plt.imshow(image_param)
plt.show()
```

## Visualize features from multiple models simultaneously

First, let's pick 2 models and specify which layers we'd want to work with

```python
from torch_dreams.model_bunch import ModelBunch

bunch = ModelBunch(
    model_dict = {
        'inception': models.inception_v3(pretrained=True).eval(),
        'resnet':    models.resnet18(pretrained= True).eval()
    }
)

layers_to_use = [
            bunch.model_dict['inception'].Mixed_6a,
            bunch.model_dict['resnet'].layer2[0].conv1
        ]

dreamy_boi = dreamer(model = bunch, quiet= False, device= 'cuda')
```

Then define a `custom_func` which determines which exact activations of the models we have to optimize

```python
def custom_func(layer_outputs):
    loss =   layer_outputs[0].mean()*2.0 + layer_outputs[1][89].mean() 
    return -loss
```

Run the optimization

```python
image_param = dreamy_boi.render(
    layers = layers_to_use,
    custom_func= custom_func,
    iters= 100
)

plt.imshow(image_param)
plt.show()
```


## Using custom transforms:

```python
import torchvision.transforms as transforms

model = models.inception_v3(pretrained=True)
dreamy_boi = dreamer(model,  device = 'cuda', quiet =  False)

my_transforms = transforms.Compose([
    transforms.RandomAffine(degrees = 10, translate = (0.5,0.5)),
    transforms.RandomHorizontalFlip(p = 0.3)
])

dreamy_boi.set_custom_transforms(transforms = my_transforms)

image_param = dreamy_boi.render(
    layers = [model.Mixed_5b],
)

plt.imshow(image_param)
plt.show()
```

## You can also use outputs of one `render()` as the input of another to create feedback loops.

```python
import matplotlib.pyplot as plt
import torchvision.models as models
from torch_dreams.dreamer import dreamer

model = models.inception_v3(pretrained=True)
dreamy_boi = dreamer(model,  device = 'cuda', quiet =  False)

image_param = dreamy_boi.render(
    layers = [model.Mixed_6c],
)

image_param = dreamy_boi.render(
    image_parameter= image_param,
    layers = [model.Mixed_5b],
    iters = 20
)

plt.imshow(image_param)
plt.show()
```

## Other conveniences 

The following methods are handy for an [`auto_image_param`](https://github.com/Mayukhdeb/torch-dreams/blob/master/torch_dreams/auto_image_param.py) instance:

1. Saving outputs as images:

```python
image_param.save('output.jpg')
```

2. Torch Tensor of dimensions `(height, width, color_channels)`

```python
torch_image = image_param.to_hwc_tensor(device = 'cpu')
```

3. Torch Tensor of dimensions `(color_channels, height, width)`

```python
torch_image_chw = image_param.to_chw_tensor(device = 'cpu')
```

4. Displaying outputs on matplotlib. 

```python
plt.imshow(image_param)
plt.show()
```

## Args for `render()`

* `layers` (`iterable`): List of the layers of model(s)'s layers to work on. `[model.layer1, model.layer2...]`
* `image_parameter` (`auto_image_param`, optional): Instance of `torch_dreams.auto_image_param.auto_image_param`
* `width` (`int`, optional): Width of image to be optimized 
* `height` (`int`, optional): Height of image to be optimized 
* `iters` (`int`, optional): Number of iterations, higher -> stronger visualization
* `lr` (`float`, optional): Learning rate
* `rotate_degrees` (`int`, optional): Max rotation in default transforms
* `scale_max` (`float`, optional): Max image size factor. Defaults to 1.1.
* `scale_min` (`float`, optional): Minimum image size factor. Defaults to 0.5.
* `translate_x` (`float`, optional): Maximum translation factor in x direction
* `translate_y` (`float`, optional): Maximum translation factor in y direction
* `custom_func` (`function`, optional): Can be used to define custom optimiziation conditions to `render()`. Defaults to None.
* `weight_decay` (`float`, optional): Weight decay for default optimizer. Helps prevent high frequency noise. Defaults to 0.
* `grad_clip` (`float`, optional): Maximum value of the norm of gradient. Defaults to 1.

## Args for `dreamer.__init__()`
 * `model` (`nn.Module` or  `torch_dreams.model_bunch.Modelbunch`): Almost any PyTorch model which was trained on imagenet `mean` and `std`, and supports variable sized images as inputs. You can pass multiple models into this argument as a `torch_dreams.model_bunch.Modelbunch` instance.
 * `quiet` (`bool`): Set to `True` if you want to disable any progress bars
 * `device` (`str`): `cuda` or `cpu` depending on your runtime 

## Acknowledgements

* [amFOSS](https://amfoss.in/)
* [Gene Kogan](https://github.com/genekogan) 

## Recommended Reading 

* [Feature Visualization](https://distill.pub/2017/feature-visualization/)
* [Google AI blog on Deepdreams](https://ai.googleblog.com/2015/06/inceptionism-going-deeper-into-neural.html)
