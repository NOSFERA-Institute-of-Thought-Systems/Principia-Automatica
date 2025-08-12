# experiments/run_pca_wavelet_experiment.py
import sys
import os
import argparse
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.decomposition import PCA

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data import data_loader
from principia_semantica import scaffolding, decomposition, utils, wavelets

@utils.timeit
def run_pca_experiment(num_scales: int, pca_components: int):
    """
    Runs a wavelet experiment with a PCA dimensionality reduction step.
    """
    print(f"\n--- Running PCA-Wavelet Experiment: scales={num_scales}, pca_components={pca_components} ---")
    
    # 1. Load Data
    embeddings, true_labels = data_loader.load_and_embed_20newsgroups()

    # 2. Scaffolding & Decomposition
    laplacian = scaffolding.construct_laplacian(embeddings, k=20)
    eigenvalues, eigenvectors = decomposition.spectral_decomposition(laplacian, m=20)

    # 3. Wavelet Feature Generation
    wavelet_features = wavelets.compute_wavelet_features(
        eigenvalues, eigenvectors, num_scales=num_scales
    )
    
    # 4. NEW STEP: Dimensionality Reduction with PCA
    print(f"Original wavelet feature dimension: {wavelet_features.shape[1]}")
    pca = PCA(n_components=pca_components, random_state=42)
    reduced_features = pca.fit_transform(wavelet_features)
    print(f"Reduced feature dimension via PCA: {reduced_features.shape[1]}")
    
    # 5. Clustering and Evaluation on REDUCED features
    kmeans = KMeans(n_clusters=20, random_state=42, n_init='auto')
    final_labels = kmeans.fit_predict(reduced_features)
    ari_score = adjusted_rand_score(true_labels, final_labels)
    
    print(f"--- Result for scales={num_scales}, pca_components={pca_components} ---")
    print(f"--> PCA-Wavelet ARI Score: {ari_score:.4f}")
    print("--- End of Experiment Run ---")
    return ari_score

def main():
    parser = argparse.ArgumentParser(description="Run PCA-on-Wavelet experiments for Principia Semantica.")
    parser.add_argument('--scales', type=int, default=10, help="Number of wavelet scales to generate.")
    parser.add_argument('--pca_dims', type=int, default=30, help="Number of dimensions to reduce to with PCA.")
    args = parser.parse_args()

    run_pca_experiment(num_scales=args.scales, pca_components=args.pca_dims)

if __name__ == "__main__":
    main()