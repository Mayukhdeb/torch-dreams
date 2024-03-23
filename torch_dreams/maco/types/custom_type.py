from typing import Callable
import torch

OperatorSignature = Callable[[torch.nn.Module, torch.Tensor, torch.Tensor], float]