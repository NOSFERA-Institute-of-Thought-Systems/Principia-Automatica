# principia_semantica/decomposition.py
import numpy as np
from scipy.sparse.linalg import eigs
from scipy.sparse import csr_matrix
from .utils import timeit, logging

@timeit
def spectral_decomposition(laplacian: csr_matrix, m: int = 20) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes the spectral decomposition of the Graph Laplacian.

    This is the core of Pillar 2. It finds the "conceptual harmonics" (eigenvectors)
    of the discourse manifold.

    Args:
        laplacian: The normalized Graph Laplacian sparse matrix.
        m: The number of eigenvectors to compute.

    Returns:
        A tuple containing:
        - eigenvalues: Sorted real eigenvalues.
        - eigenvectors: Corresponding real eigenvectors.
    """
    if m >= laplacian.shape[0]:
        raise ValueError(f"Number of eigenvectors m ({m}) must be smaller than the number of documents ({laplacian.shape[0]}).")

    logging.info(f"Computing the {m} smallest eigenvectors using SciPy's 'eigs'.")
    # 'SM' computes the eigenvalues with the smallest magnitude.
    eigenvalues, eigenvectors = eigs(laplacian, k=m, which='SM')

    # Sort the results
    order = np.argsort(np.real(eigenvalues))
    eigenvalues, eigenvectors = np.real(eigenvalues[order]), np.real(eigenvectors[:, order])

    logging.info(f"Smallest 5 eigenvalues: {eigenvalues[:5]}")
    return eigenvalues, eigenvectors

