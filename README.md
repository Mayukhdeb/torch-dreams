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
* [Caricatures](https://github.com/Mayukhdeb/torch-dreams#caricatures)
* [Visualize features from multiple models simultaneously](https://github.com/Mayukhdeb/torch-dreams#visualize-features-from-multiple-models-simultaneously)
* [Use custom transforms](https://github.com/Mayukhdeb/torch-dreams#using-custom-transforms)
* [Feedback loops](https://github.com/Mayukhdeb/torch-dreams#you-can-also-use-outputs-of-one-render-as-the-input-of-another-to-create-feedback-loops)
* [Custom images](https://github.com/Mayukhdeb/torch-dreams#using-custom-images)
* [Masked image parametrs](https://github.com/Mayukhdeb/torch-dreams#masked-image-parameters)
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
## Caricatures

Caricatures create a new image that has a similar but more extreme activation pattern to the input image at a given layer (or multiple layers at a time). It's inspired from [this issue](https://github.com/tensorflow/lucid/issues/121)

<img src = "https://raw.githubusercontent.com/Mayukhdeb/torch-dreams/master/images/caricature.png" width = "70%">

In this case, let's use googlenet 

```python
model = models.googlenet(pretrained = True)
dreamy_boi = dreamer(model = model, quiet= False, device= 'cuda')

image_param = dreamy_boi.caricature(
    input_tensor = image_tensor, 
    layers = [model.inception4c],   ## feel free to append more layers for more interesting caricatures 
    power= 1.2,                     ## higher -> more "exaggerated" features
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

## Using custom images

Note that you might have to use smaller values for certain hyperparameters like `lr` and `grad_clip`.

```python
from torch_dreams.custom_image_param import custom_image_param
param = custom_image_param(filename = 'images/sample_small.jpg', device= 'cuda')

image_param = dreamy_boi.render(
    image_parameter= param,
    layers = [model.Mixed_6c],
    lr = 2e-4,
    grad_clip = 0.1,
    weight_decay= 1e-1,
    iters = 120
)
```

## Masked Image parameters

Can be used to optimize only certain parts of the image using a mask whose values are clipped between `[0,1]`.

<img src = "https://raw.githubusercontent.com/Mayukhdeb/torch-dreams/master/images/masked_param.png" width = "80%">

Here's an example with a vertical gradient 

```python 
from torch_dreams.masked_image_param import masked_image_param

mask = torch.ones(1,1,512,512)

for i in range(0, 512, 1):  ## vertical gradient
    mask[:,:,i,:] = (i/512)

param = masked_image_param(
    image= 'images/sample_small.jpg',  ## optional
    mask_tensor = mask,
    device = 'cuda'
)

param = dreamy_boi.render(
    layers = [model.inception4c],
    image_parameter= param,
    lr = 1e-4,
    grad_clip= 0.1,
    weight_decay= 1e-1,
    iters= 200,
)

param.save('masked_param_output.jpg')
```

It's also possible to update the mask on the fly with `param.update_mask(some_mask)`

```python

param.update_mask(mask = torch.flip(mask, dims = (2,)))

param = dreamy_boi.render(
    layers = [model.inception4a],
    image_parameter= param,
    lr = 1e-4,
    grad_clip= 0.1,
    weight_decay= 1e-1,
    iters= 200,
)

param.save('masked_param_output_2.jpg')
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

5. For instances of `custom_image_param`, you can set any NCHW tensor as the image parameter: 

```python
image_tensor = image_param.to_nchw_tensor()

## do some stuff with image_tensor
t = transforms.Compose([
    transforms.RandomRotation(5)
])
transformed_image_tensor = t(image_tensor) 

image_param.set_param(tensor = transformed_image_tensor)
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
