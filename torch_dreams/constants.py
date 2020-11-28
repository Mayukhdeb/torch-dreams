
import numpy as np
import torch


IMAGENET_MEAN_1 = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD_1 = np.array([0.229, 0.224, 0.225], dtype=np.float32)


IMAGENET_MEAN_1_GRAY = np.array([0.449], dtype=np.float32)
IMAGENET_STD_1_GRAY = np.array([0.226], dtype=np.float32)

device = "cuda" if torch.cuda.is_available() else 'cpu'

LOWER_IMAGE_BOUND = torch.tensor((-IMAGENET_MEAN_1 / IMAGENET_STD_1).reshape(1, -1, 1, 1)).to(device)
UPPER_IMAGE_BOUND = torch.tensor(((1 - IMAGENET_MEAN_1) / IMAGENET_STD_1).reshape(1, -1, 1, 1)).to(device)
KERNEL_SIZE = 9  # "magic number" picked this one as it just works well
