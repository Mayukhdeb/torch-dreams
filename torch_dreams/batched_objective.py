from typing import Callable, List

import torch


class BatchedObjective:
    def __init__(self, objectives: List[Callable]) -> None:
        """Wrapper to handle generations of multiple feature visualizations batch-wise.

        Args:
            objectives (List[Callable]): list of all custom_func i.e objective functions. It's length should be same as that of the batch size.
        """
        self.objectives = objectives

    def __call__(self, x):
        assert torch.is_tensor(
            x
        ), f"Expected x to be a torch.tensor, but got: {type(x)}"

        batch_size = x.shape[0]

        assert batch_size == len(
            self.objectives
        ), f"Batch size of x and len(self.objectives) do not match: {x.shape[0]} and {len(self.objectives)}"

        loss_sum = torch.tensor(0.0)

        for i in range(batch_size):
            loss_sum += self.objectives[i](
                x[i].unsqueeze(0)
            )  ## is unsqueeze this really necessary? idk

        return loss_sum  ## then call loss_sum.backward()
