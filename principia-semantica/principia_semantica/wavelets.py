# principia_semantica/wavelets.py
import numpy as np
from .utils import timeit, logging

@timeit
def compute_wavelet_features(
    eigenvalues: np.ndarray, 
    eigenvectors: np.ndarray, 
    num_scales: int = 4
) -> np.ndarray:
    """
    Computes Spectral Graph Wavelet features from the Laplacian's eigensystem.

    This function operationalizes Proposal 4.1. Instead of using the eigenvectors
    directly, it uses them as a basis to compute wavelet coefficients at
    multiple scales. This provides a multi-resolutional view of each node's
    neighborhood.

    Args:
        eigenvalues: The sorted eigenvalues of the Graph Laplacian.
        eigenvectors: The corresponding eigenvectors (as columns).
        num_scales: The number of wavelet scales to compute. A key hyperparameter.

    Returns:
        A (n_documents, m_eigenvectors * num_scales) numpy array of wavelet features.
    """
    logging.info(f"Computing wavelet features for {num_scales} scales.")
    
    # Define the scales. We use a logarithmic scale to cover different resolutions.
    # The max eigenvalue is theoretically bounded by 2 for a normalized Laplacian.
    s_min = 2 / eigenvalues[-1] if eigenvalues[-1] > 0 else 1
    s_max = 20 / eigenvalues[-1] if eigenvalues[-1] > 0 else 20
    scales = np.geomspace(s_min, s_max, num=num_scales)

    # The wavelet kernel g(sλ). We use a simple heat kernel: e^(-sλ).
    # This is applied to the eigenvalues for each scale.
    # Shape: (num_scales, num_eigenvectors)
    g_s_lambda = np.exp(-scales[:, np.newaxis] @ eigenvalues[np.newaxis, :])

    # The wavelet coefficients for all nodes are computed efficiently as:
    # W_s = U * g(sΛ) * U^T
    # where U is the eigenvector matrix and g(sΛ) is a diagonal matrix.
    # We compute this for each scale and concatenate the results.

    all_wavelet_features = []
    for i in range(num_scales):
        # g_s_lambda[i, :] represents the diagonal of g(sΛ) for the i-th scale
        # We compute U * diag(g_s_lambda[i, :])
        wavelet_operator_at_scale_s = eigenvectors * g_s_lambda[i, :]
        
        # Then we multiply by U^T to get the full set of features at this scale
        # This gives us a (num_docs, num_docs) matrix where column j is the
        # wavelet coefficients for node j.
        # However, a more stable feature representation is simply U*g(sΛ).
        # We will use this (num_docs, num_eigenvectors) representation for each scale.
        features_at_scale_s = wavelet_operator_at_scale_s
        all_wavelet_features.append(features_at_scale_s)

    # Concatenate features from all scales side-by-side
    # Final shape: (n_docs, n_eigenvectors * num_scales)
    final_feature_matrix = np.concatenate(all_wavelet_features, axis=1)
    
    logging.info(f"Wavelet feature matrix created with shape {final_feature_matrix.shape}")
    return final_feature_matrix