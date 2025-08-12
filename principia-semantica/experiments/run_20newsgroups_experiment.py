# experiments/run_20newsgroups_experiment.py
import sys
import os
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data import data_loader
from principia_semantica import scaffolding, decomposition, utils, visualizer, wavelets # <-- ADD wavelets

@utils.timeit
def main():
    """
    Runs and compares standard spectral clustering vs. wavelet-based spectral clustering.
    """
    print("\n--- Starting Principia Semantica 20 Newsgroups Experiment (Wavelet Comparison) ---")
    
    # 1. Load Data
    embeddings, true_labels = data_loader.load_and_embed_20newsgroups()

    # 2. Run Pillar 1: Scaffolding
    laplacian = scaffolding.construct_laplacian(embeddings, k=20)
    
    # 3. Run Pillar 2: Decomposition
    num_eigenvectors = 20
    eigenvalues, eigenvectors = decomposition.spectral_decomposition(laplacian, m=num_eigenvectors)

    # --- 4a. Baseline: Standard Spectral Clustering ---
    print("\n--- Running Baseline: Standard Spectral Clustering ---")
    spectral_embedding = eigenvectors[:, 1:num_eigenvectors] # Skip v0
    kmeans_spectral = KMeans(n_clusters=20, random_state=42, n_init='auto')
    spectral_labels = kmeans_spectral.fit_predict(spectral_embedding)
    ari_spectral = adjusted_rand_score(true_labels, spectral_labels)
    
    # --- 4b. Hypothesis Test: Wavelet-based Clustering ---
    print("\n--- Running Hypothesis Test: Wavelet-Adaptive Clustering ---")
    # Generate features using our new module
    wavelet_features = wavelets.compute_wavelet_features(
        eigenvalues, 
        eigenvectors, 
        num_scales=5 # A good starting hyperparameter
    )
    kmeans_wavelet = KMeans(n_clusters=20, random_state=42, n_init='auto')
    wavelet_labels = kmeans_wavelet.fit_predict(wavelet_features)
    ari_wavelet = adjusted_rand_score(true_labels, wavelet_labels)
    
    # --- 5. Report and Visualize Results ---
    print("\n--- Final Results ---")
    print(f"Baseline (Eigenvector) ARI Score: {ari_spectral:.4f}")
    print(f"Hypothesis (Wavelet) ARI Score:   {ari_wavelet:.4f}")

    if ari_wavelet > ari_spectral:
        print("\nHypothesis Confirmed: Wavelet features provide a superior basis for clustering.")
    else:
        print("\nHypothesis Not Confirmed: Further investigation needed.")

    results_dir = "results/20_newsgroups"
    visualizer.visualize_clusters(
        embeddings, 
        true_labels, 
        title="20 Newsgroups Ground Truth",
        output_dir=results_dir
    )
    visualizer.visualize_clusters(
        embeddings, 
        spectral_labels, 
        title=f"Discovered Clusters (Eigenvectors) - ARI {ari_spectral:.3f}",
        output_dir=results_dir
    )
    visualizer.visualize_clusters(
        embeddings, 
        wavelet_labels, 
        title=f"Discovered Clusters (Wavelets) - ARI {ari_wavelet:.3f}",
        output_dir=results_dir
    )
    
    print("\n--- 20 Newsgroups Experiment Complete ---")

if __name__ == "__main__":
    main()