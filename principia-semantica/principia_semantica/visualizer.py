# principia_semantica/visualizer.py
import numpy as np
import matplotlib.pyplot as plt
import umap
import os

from .utils import timeit, logging

@timeit
def visualize_clusters(
    embeddings: np.ndarray,
    labels: np.ndarray,
    title: str,
    output_dir: str = "results",
    is_continuous: bool = False
):
    """
    Reduces embeddings to 2D using UMAP and creates a scatter plot of the clusters.

    Args:
        embeddings: The high-dimensional document embeddings.
        labels: The integer (discrete) or float (continuous) labels for each document.
        title: The title for the plot.
        output_dir: Directory to save the plot image.
        is_continuous: If True, treats labels as continuous values for a color bar.
    """
    logging.info(f"Generating UMAP visualization: '{title}'")
    os.makedirs(output_dir, exist_ok=True)
    
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)
    embedding_2d = reducer.fit_transform(embeddings)
    
    plt.figure(figsize=(16, 12))
    
    cmap = 'viridis' if is_continuous else 'Spectral'
    
    scatter = plt.scatter(
        embedding_2d[:, 0], 
        embedding_2d[:, 1], 
        c=labels,
        cmap=cmap, 
        s=5,
        alpha=0.7
    )
    
    if is_continuous:
        colorbar = plt.colorbar(scatter)
        colorbar.set_label("Ricci Curvature", fontsize=12)
    
    plt.title(title, fontsize=18)
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    plt.gca().set_aspect('equal', 'datalim')
    
    filename = title.lower().replace(" ", "_").replace("(", "").replace(")", "").replace(".", "") + ".png"
    output_path = os.path.join(output_dir, filename)
    
    plt.savefig(output_path)
    logging.info(f"Plot saved to {output_path}")
    plt.close()