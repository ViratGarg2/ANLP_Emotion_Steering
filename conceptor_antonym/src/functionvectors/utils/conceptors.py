import numpy as np
import torch
from tqdm import tqdm
from typing import Dict, List, Optional


def scale_conceptor(C, aperture_scaling_factor):
    return C @ torch.inverse(C + (aperture_scaling_factor**-2) * (torch.eye(C.shape[0], device=C.device) - C))


def rescale_conceptor(C, prev_alpha, new_alpha):
    return scale_conceptor(C, new_alpha / prev_alpha)


def compute_conceptor(X, aperture):
    """
    Computes the conceptor matrix for a given input matrix X.
    (PyTorch version)

    Parameters:
    - X (torch.Tensor): Input matrix of shape (n_samples, n_features).
    - torch.Tensor: Conceptor matrix of shape (n_features, n_features).
    """
    R = torch.matmul(X.T, X) / X.shape[0]
    U, S, _ = torch.svd(R)
    C = U * (S / (S + (aperture ** (-2)) * torch.ones(S.shape, device=X.device))) @ U.T
    return C


def combine_conceptors_and(C1, C2):
    """
    Combines two conceptors C1 and C2 using the given formula. (AND operation, does not work so well)

    Parameters:
    - C1 (torch.Tensor): First conceptor tensor of shape (n_features, n_features).
    - C2 (torch.Tensor): Second conceptor tensor of shape (n_features, n_features).

    Returns:
    - torch.Tensor: Combined conceptor tensor of shape (n_features, n_features).
    """
    I = torch.eye(C1.shape[0], device=C1.device)  # Identity matrix
    C1_inv = torch.inverse(C1)
    C2_inv = torch.inverse(C2)
    combined_inv = C1_inv + C2_inv - I
    combined = torch.inverse(combined_inv)
    return combined


def combine_conceptors(C1, C2):
    """
    TODO: what is this?
    Combines two conceptors C1 and C2 using the given new formula. (OR operation which works much better than AND)

    Parameters:
    - C1 (torch.Tensor): First conceptor tensor of shape (n_features, n_features).
    - C2 (torch.Tensor): Second conceptor tensor of shape (n_features, n_features).

    Returns:
    - torch.Tensor: Combined conceptor tensor of shape (n_features, n_features).
    """
    I = torch.eye(C1.shape[0], device=C1.device)  # Identity matrix
    I_C1_inv = torch.inverse(I - C1)
    I_C2_inv = torch.inverse(I - C2)
    combined_inv = I_C1_inv + I_C2_inv - I
    combined = torch.inverse(combined_inv)
    result = I - combined
    return result


def cache_conceptors(
    activations_cache: Dict[int, Dict[int, np.ndarray]],
    list_extraction_layers: List[int],
    list_apertures: List[float],
    num_experiments: int,
    mean_train_activations: Optional[Dict[int, np.ndarray]] = None,
    use_conceptor_rescaling: bool = True,
):
    # Initialize a dictionary to store conceptors for each experiment
    conceptors_cache = {exp: {} for exp in range(num_experiments)}

    # Total number of computations for the progress bar
    total_computations = len(list_extraction_layers) * num_experiments
    if not use_conceptor_rescaling:
        total_computations *= len(list_apertures)

    # Precompute conceptors for all layers and apertures with a progress bar
    with tqdm(total=total_computations, desc="Precomputing conceptors") as pbar:
        for exp in range(num_experiments):
            for layer in list_extraction_layers:
                if use_conceptor_rescaling:
                    ### first aperture
                    aperture = sorted(list_apertures)[len(list_apertures) // 2]
                    # Extract the last-token activations of steering examples at the specified layer
                    activations = activations_cache[exp][layer]
                    # Apply mean centering if mean_train_activations is provided
                    if mean_train_activations is not None:
                        activations = activations - mean_train_activations[layer].to(activations.device)
                    # Compute the steering conceptor using cached activations
                    ### this runs faster on cpu
                    conceptor = compute_conceptor(activations.cpu(), aperture)
                    # Store the conceptor in the cache
                    conceptors_cache[exp][(layer, aperture)] = conceptor.cpu()
                    ### other apertures
                    for new_aperture in list_apertures:
                        if new_aperture != aperture:
                            ### scale the conceptor (this runs faster on GPU)
                            new_conceptor = rescale_conceptor(conceptor.cuda(), aperture, new_aperture)
                            conceptors_cache[exp][(layer, new_aperture)] = new_conceptor.detach().cpu()
                    # update the progress bar
                    pbar.update(1)
                else:
                    for aperture in list_apertures:
                        # Extract the last-token activations of steering examples at the specified layer
                        activations = activations_cache[exp][layer]
                        # Apply mean centering if mean_train_activations is provided
                        if mean_train_activations is not None:
                            activations = activations - mean_train_activations[layer].to(activations.device)
                        # Compute the steering conceptor using cached activations
                        conceptor = compute_conceptor(activations, aperture)
                        # Store the conceptor in the cache
                        conceptors_cache[exp][(layer, aperture)] = conceptor.detach().cpu()
                    # Update the progress bar
                    pbar.update(1)

    return conceptors_cache
