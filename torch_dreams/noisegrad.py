import torch
import torch.nn as nn

import copy


class NoiseGradModule(nn.Module):
    """
    A tiny wrapper to apply noisegrad in nn.Module instances.

    Original paper: https://arxiv.org/pdf/2106.10185.pdf

    Original Implementation: https://github.com/understandable-machine-intelligence-lab/NoiseGrad/blob/master/src/noisegrad.py

    Args:
        module (nn.Module): a layer/module within the model
        mean (float): mean of the distribution from which we'd sample the noise values
        std (float): standard deviation of the distribution from which we'd sample the noise values

    Usage:

    ```python
    import torchvision.models as models
    from torch_dreams.noisegrad import NoiseGradModule

    model = models.inception_v3(pretrained=True)
    model.Mixed_5b = NoiseGradModule(module = model.Mixed_5b, mean = 1, std = 0.2)
    ```
    """

    def __init__(self, module: nn.Module, mean: float = 1, std: float = 0.2):
        super().__init__()

        self.mean = mean
        self.std = std

        # Creates a normal (also called Gaussian) distribution.
        self.distribution = torch.distributions.normal.Normal(
            loc=self.mean, scale=self.std
        )

        self.module = module

    @torch.no_grad()
    def sample(self):
        noisy_module = copy.deepcopy(self.module)

        for layer in noisy_module.parameters():
            noise = self.distribution.sample(layer.size()).to(layer.device)
            layer.mul_(noise)

        return noisy_module

    def forward(self, x):
        sampled_noisy_layer = self.sample()
        return sampled_noisy_layer.forward(x)
