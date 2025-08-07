# paper-01-spectral-geometry/scripts/paper1_visualize.py

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import fetch_20newsgroups
import umap
import os

def main():
    """
    Final visualization pipeline for Paper 1.
    1. Load ground truth labels for comparison.
    2. Load the pre-computed spectral embedding.
    3. Perform k-means clustering on the spectral embedding.
    4. Generate a UMAP of the original embeddings, colored by our new cluster labels.
    """
    print("--- Starting Paper 1 Visualization ---")

    # --- 1. Load Ground Truth Labels & UMAP Data ---
    print("Loading original data for comparison and UMAP...")
    newsgroups = fetch_20newsgroups(subset='all')
    true_labels = newsgroups.target
    # We reload the high-dim embeddings to create the 2D UMAP projection
    high_dim_embeddings = np.load("paper-01-spectral-geometry/data/embeddings.npy")

    # --- 2. Load Spectral Embedding ---
    results_path = "paper-01-spectral-geometry/results/spectral_embedding.npy"
    print(f"Loading spectral embedding from {results_path}...")
    spectral_embedding = np.load(results_path)
    
    # The first eigenvector (for eigenvalue 0) is constant and not useful for clustering.
    # We use eigenvectors 1 through k.
    embedding_for_clustering = spectral_embedding[:, 1:16]

    # --- 3. Perform K-Means Clustering ---
    n_clusters = 20 # We know there are 20 newsgroups
    print(f"\nPerforming K-Means clustering with {n_clusters} clusters...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    discovered_labels = kmeans.fit_predict(embedding_for_clustering)
    print("-> Clustering complete.")

    # --- 4. Generate UMAP Visualization ---
    print("\nGenerating final UMAP plot colored by discovered clusters...")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42, metric='cosine')
    embedding_2d = reducer.fit_transform(high_dim_embeddings)

    plt.figure(figsize=(16, 12))
    plt.scatter(
        embedding_2d[:, 0],
        embedding_2d[:, 1],
        c=discovered_labels, # COLOR BY OUR DISCOVERED LABELS
        cmap='Spectral',
        s=5,
        alpha=0.7
    )
    plt.title('UMAP Projection colored by Spectral Clustering Labels', fontsize=18)
    plt.xlabel('UMAP Dimension 1', fontsize=12)
    plt.ylabel('UMAP Dimension 2', fontsize=12)
    plt.gca().set_aspect('equal', 'datalim')
    
    # Save the figure
    output_dir = "paper-01-spectral-geometry/results"
    output_path = os.path.join(output_dir, 'umap_spectral_clusters.png')
    plt.savefig(output_path)
    print(f"-> Plot saved to {output_path}")
    print("\n--- Visualization complete. ---")


if __name__ == "__main__":
    main()