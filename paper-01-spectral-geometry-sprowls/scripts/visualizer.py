# paper-01-spectral-geometry-sprowls/scripts/visualizer.py
import numpy as np
import matplotlib.pyplot as plt
import umap

def visualize_clusters(high_dim_embeddings, labels, title, output_path):
    print(f"\nGenerating UMAP plot: {title}...")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42, metric='cosine')
    embedding_2d = reducer.fit_transform(high_dim_embeddings)
    
    plt.figure(figsize=(16, 12))
    plt.scatter(
        embedding_2d[:, 0], embedding_2d[:, 1], c=labels,
        cmap='Spectral', s=5, alpha=0.7
    )
    plt.title(title, fontsize=18)
    plt.xlabel('UMAP Dimension 1', fontsize=12)
    plt.ylabel('UMAP Dimension 2', fontsize=12)
    plt.gca().set_aspect('equal', 'datalim')
    
    plt.savefig(output_path)
    print(f"-> Plot saved to {output_path}")
    plt.close()