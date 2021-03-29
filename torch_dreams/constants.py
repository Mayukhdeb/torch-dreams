import numpy as np 
import torch

def color_correlation_normalized():
    color_correlation_svd_sqrt = np.asarray([[0.26, 0.09, 0.02],
                                             [0.27, 0.00, -0.05],
                                             [0.27, -0.09, 0.03]]).astype(np.float32)
    max_norm_svd_sqrt = np.max(np.linalg.norm(color_correlation_svd_sqrt, axis=0))
    color_correlation_normalized = torch.tensor(color_correlation_svd_sqrt / max_norm_svd_sqrt)
    return color_correlation_normalized

def imagenet_mean_std():
    return (torch.tensor([0.485, 0.456, 0.406]), 
            torch.tensor([0.229, 0.224, 0.225]))


class Constants:
    color_correlation_matrix = color_correlation_normalized()
    imagenet_mean, imagenet_std = imagenet_mean_std()

