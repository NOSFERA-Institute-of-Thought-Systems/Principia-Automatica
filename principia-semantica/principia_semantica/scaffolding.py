# principia_semantica/scaffolding.py
import numpy as np
import networkx as nx
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix, eye as sparse_eye

from .utils import timeit, logging

@timeit
def construct_laplacian(
    embeddings: np.ndarray, 
    k: int = 15,
    return_graph: bool = False
) -> tuple | csr_matrix:
    """
    Constructs the normalized Graph Laplacian from document embeddings.

    This function represents the core of Pillar 1: Geometric Scaffolding.
    It approximates the manifold with a k-NN graph and computes its Laplacian,
    which is a discrete analog of the Laplace-Beltrami operator.

    Args:
        embeddings: A (n_documents, n_dimensions) numpy array of document embeddings.
        k: The number of nearest neighbors to use for graph construction.
        return_graph: If True, also returns the NetworkX graph object.

    Returns:
        The normalized Graph Laplacian as a SciPy sparse matrix.
        If return_graph is True, returns a tuple (laplacian, graph).
    """
    n_docs, _ = embeddings.shape
    if n_docs < k + 1:
        raise ValueError(f"Number of documents ({n_docs}) must be greater than k ({k}).")

    logging.info(f"Constructing k-NN graph with k={k} for {n_docs} documents.")
    nn = NearestNeighbors(n_neighbors=k + 1, metric='cosine', algorithm='brute', n_jobs=-1)
    nn.fit(embeddings)
    _, indices = nn.kneighbors(embeddings)

    logging.info("Building adjacency matrix.")
    rows = np.arange(n_docs).repeat(k)
    cols = indices[:, 1:].flatten()
    data = np.ones(len(rows))
    
    adjacency = csr_matrix((data, (rows, cols)), shape=(n_docs, n_docs))
    adjacency = adjacency + adjacency.T
    adjacency.data = np.ones(len(adjacency.data))

    logging.info("Calculating normalized Laplacian.")
    degree = np.array(adjacency.sum(axis=1)).flatten()
    
    non_zero_degree_indices = np.where(degree > 0)[0]
    if len(non_zero_degree_indices) == 0:
        raise ValueError("Graph has no edges. Try increasing k.")
    
    degree_inv_sqrt = 1.0 / np.sqrt(degree[non_zero_degree_indices])
    D_inv_sqrt = csr_matrix((degree_inv_sqrt, (non_zero_degree_indices, non_zero_degree_indices)), shape=(n_docs, n_docs))
    laplacian = sparse_eye(n_docs, format='csr') - D_inv_sqrt @ adjacency @ D_inv_sqrt
    
    if return_graph:
        logging.info("Building NetworkX graph for curvature analysis.")
        G = nx.from_scipy_sparse_array(adjacency)
        return laplacian, G
    
    return laplacian