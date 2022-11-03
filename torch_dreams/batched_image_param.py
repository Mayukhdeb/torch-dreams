import torch
from typing import List
from .auto_image_param import BaseImageParam


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
