# experiments/run_ablation_study.py
import sys
import os
import argparse
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data import data_loader
from principia_semantica import scaffolding, decomposition, utils, wavelets

@utils.timeit
def run_single_ablation(k_neighbors: int, num_eigenvectors: int, num_scales: int):
    """Runs a single configuration of the wavelet experiment."""
    print(f"\n--- Running Ablation: k={k_neighbors}, m={num_eigenvectors}, scales={num_scales} ---")
    
    # 1. Load Data (uses cache)
    embeddings, true_labels = data_loader.load_and_embed_20newsgroups()

    # 2. Scaffolding
    laplacian = scaffolding.construct_laplacian(embeddings, k=k_neighbors)
    
    # 3. Decomposition
    eigenvalues, eigenvectors = decomposition.spectral_decomposition(laplacian, m=num_eigenvectors)

    # 4. Wavelet Feature Generation
    wavelet_features = wavelets.compute_wavelet_features(
        eigenvalues, eigenvectors, num_scales=num_scales
    )
    
    # 5. Clustering and Evaluation
    kmeans = KMeans(n_clusters=20, random_state=42, n_init='auto')
    wavelet_labels = kmeans.fit_predict(wavelet_features)
    ari_score = adjusted_rand_score(true_labels, wavelet_labels)
    
    print(f"--- Result for k={k_neighbors}, m={num_eigenvectors}, scales={num_scales} ---")
    print(f"--> Wavelet ARI Score: {ari_score:.4f}")
    print("--- End of Ablation Run ---")
    return ari_score

def main():
    parser = argparse.ArgumentParser(description="Run ablation studies for Principia Semantica.")
    parser.add_argument('--k', type=int, default=20, help="Number of nearest neighbors.")
    parser.add_argument('--m', type=int, default=20, help="Number of eigenvectors.")
    parser.add_argument('--scales', type=int, default=5, help="Number of wavelet scales.")
    args = parser.parse_args()

    run_single_ablation(k_neighbors=args.k, num_eigenvectors=args.m, num_scales=args.scales)

if __name__ == "__main__":
    main()