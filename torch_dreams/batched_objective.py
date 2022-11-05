from typing import Callable, List

import torch


class BatchedObjective:
    def __init__(self, objectives: List[Callable]) -> None:
        """Wrapper to handle generations of multiple feature visualizations batch-wise.

        Args:
            objectives (List[Callable]): list of all custom_func i.e objective functions. It's length should be same as that of the batch size.
        """
        self.objectives = objectives

    def __call__(self, x: list):

        for y in x:
            assert torch.is_tensor(
                y
            ), f"Expected every item in x to be a torch.tensor, but got: {type(y)}"

        batch_size = x[0].shape[0]

        assert batch_size == len(
            self.objectives
        ), f"Batch size of x and len(self.objectives) do not match: {x[0].shape[0]} and {len(self.objectives)}"

        loss_sum = torch.tensor(0.0).to(x[0].device)

        for batch_idx in range(batch_size):
            loss_sum += self.objectives[batch_idx](
                x[0][batch_idx].unsqueeze(0)
            )  ## is unsqueeze this really necessary? idk

        return loss_sum  ## then call loss_sum.backward()
