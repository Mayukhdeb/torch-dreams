import torch
from typing import List
from .auto_image_param import BaseImageParam, AutoImageParam


class BatchedOptimizer:
    def __init__(self, optimizers: List) -> None:
        """Thin abstraction layer to make sure the top level stuff does not change in torch_dreams.Dreamer().render()

        Args:
            optimizers (List): list of torch optimizers
        """
        self.optimizers = optimizers

    def zero_grad(self):
        for opt in self.optimizers:
            opt.zero_grad()

    def step(self):
        for opt in self.optimizers:
            opt.step()


class BatchedImageParam:
    def __init__(
        self,
        image_params: List[BaseImageParam],
    ) -> None:
        self.image_params = image_params

        ## Making sure all of them have the same width and height
        for x in self.image_params:
            assert (
                x.width == self.image_params[0].width
            ), f"Expected widths of all image_params to match, but got: {[z.width for z in self.image_params]}"
            assert (
                x.height == self.image_params[0].height
            ), f"Expected heights of all image_params to match, but got: {[z.height for z in self.image_params]}"

    def forward(self, device: str):
        image_param_batch = torch.cat(
            [x.forward(device=device) for x in self.image_params]
        )
        return image_param_batch

    def clip_grads(self, grad_clip: float = 1.0):
        for p in self.image_params:
            p.clip_grads(grad_clip=grad_clip)


class BatchedAutoImageParam(BatchedImageParam):
    def __init__(
        self,
        batch_size: int = 1,
        height: int = 256,
        width: int = 256,
        standard_deviation: float = 0.01,
        device: str = "cuda:0",
        weight_decay=0.0,
        lr=9e-3,
    ) -> None:
        image_params = [
            AutoImageParam(
                height=height,
                width=width,
                standard_deviation=standard_deviation,
                device=device,
            )
            for x in range(batch_size)
        ]
        for p in image_params:
            p.get_optimizer(lr=lr, weight_decay=weight_decay)

        self.optimizer = BatchedOptimizer(
            optimizers=[x.optimizer for x in image_params]
        )
        super().__init__(image_params)

    def __getitem__(self, idx):
        return self.image_params[idx]

    def __len__(self):
        return len(self.image_params)
