# Torch-Dreams
Making deep neural networks more interpretable, for research, art and sometimes both simultaneously. 

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Mayukhdeb/torch-dreams-notebooks/blob/main/notebooks/torch_dreams_starter.ipynb)
[![](https://img.shields.io/github/last-commit/mayukhdeb/torch-dreams)](https://github.com/mayukhdeb/torch-dreams/commits/master)

<img src = "https://raw.githubusercontent.com/Mayukhdeb/torch-dreams/master/images/3_grad_masks.jpg">

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


## Important links:

* [Feature visualization by Olah, et al.](https://distill.pub/2017/feature-visualization/)
* [Google AI blog on DeepDreams](https://ai.googleblog.com/2015/06/inceptionism-going-deeper-into-neural.html)


