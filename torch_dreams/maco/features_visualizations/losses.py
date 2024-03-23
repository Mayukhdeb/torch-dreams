import torch


# implement the cosine_similarity function
def cosine_similarity(tensor_a: torch.Tensor, tensor_b: torch.Tensor) -> torch.Tensor:
    """
    Calculate the cosine similarity between two tensors.

    Args:
        tensor_a (torch.Tensor): The first tensor.
        tensor_b (torch.Tensor): The second tensor.

    Returns:
        torch.Tensor: The cosine similarity between the two tensors.
    """
    axis_to_norm = tuple(range(1, len(tensor_a.shape)))

    tensor_a = torch.nn.functional.normalize(tensor_a, dim = axis_to_norm)
    tensor_b = torch.nn.functional.normalize(tensor_b, dim = axis_to_norm)

    return torch.sum(tensor_a * tensor_b, axis = axis_to_norm)





def dot_cossim(tensor_a: torch.Tensor, tensor_b: torch.Tensor,cossim_pow: float = 2.0) -> torch.Tensor:
    """
    Return the product of the cosine similarity and the dot product for batches of vectors passed.
    This original looking loss was proposed by the authors of lucid and seeks to both optimise
    the direction with cosine similarity, but at the same time exaggerate the feature (caricature)
    with the dot product.

    Parameters
    ----------
    tensor_a
        Batch of N tensors.
    tensor_b
        Batch of N tensors.
    cossim_pow
        Power of the cosine similarity, higher value encourage the objective to care more about
        the angle of the activations.

    Returns
    -------
    dot_cossim_value
        The product of the cosine similarity and the dot product for each pairs of tensors.

    """

    cosim = torch.pow(torch.clamp_min(cosine_similarity(tensor_a, tensor_b), 1e-1), cossim_pow)
    dot = torch.sum(tensor_a * tensor_b)

    return cosim * dot


