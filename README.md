# Torch-Dreams
Making neural networks more interpretable, for research and art. 


[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Mayukhdeb/torch-dreams-notebooks/blob/main/docs_notebooks/hello_torch_dreams.ipynb)
[![build](https://github.com/Mayukhdeb/torch-dreams/actions/workflows/main.yml/badge.svg)](https://github.com/Mayukhdeb/torch-dreams/actions/workflows/main.yml)
[![](https://img.shields.io/github/last-commit/mayukhdeb/torch-dreams)](https://github.com/mayukhdeb/torch-dreams/commits/master)
[![](https://img.shields.io/twitter/url?label=Docs&style=flat-square&url=https%3A%2F%2Fapp.gitbook.com%2F%40mayukh09%2Fs%2Ftorch-dreams%2F)](https://app.gitbook.com/@mayukh09/s/torch-dreams/)


[<img src = "images/banner_segmentation_model.png">](https://app.gitbook.com/@mayukh09/s/torch-dreams/visualizing-individual-channels)

```
pip install torch-dreams 
```

## Quick start
> Make sure you also check out the [quick start colab notebook](https://colab.research.google.com/github/Mayukhdeb/torch-dreams-notebooks/blob/main/docs_notebooks/hello_torch_dreams.ipynb) and the [docs](https://app.gitbook.com/@mayukh09/s/torch-dreams/) for more interesting examples. 


```python
import matplotlib.pyplot as plt
import torchvision.models as models
from torch_dreams.dreamer import dreamer

model = models.inception_v3(pretrained=True)
dreamy_boi = dreamer(model,  device = 'cuda', quiet =  False)

image_param = dreamy_boi.render(
    layers = [model.Mixed_5b],
)

plt.imshow(image_param.rgb.astype(np.float32))
plt.show()
```

## You can also optimize activations from multiple models simultaneously

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

dreamy_boi = dreamer(model = bunch, quiet= True, device= 'cuda')
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

plt.imshow((image_param.rgb).astype(np.float32))
plt.show()
```


---

## Visualize individual channels

```python

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
plt.imshow(image_param.rgb.astype(np.float32))
plt.show()
```

## Acknowledgements

* [amFOSS](https://amfoss.in/)
* [Gene Kogan](https://github.com/genekogan) 

