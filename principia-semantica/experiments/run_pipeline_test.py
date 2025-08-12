# experiments/run_pipeline_test.py
import numpy as np
from sklearn.cluster import KMeans
import sys
import os

# Add the parent directory to the path to find our library
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from principia_semantica import scaffolding, decomposition, analysis, utils

@utils.timeit
def main():
    """A full pipeline test on dummy data."""
    print("\n--- Starting Principia Semantica Pipeline Test ---")
    
    # 1. Generate dummy data
    print("Generating dummy data: 3 distinct clusters.")
    n_samples = 300
    n_features = 50
    centers = [[1, 1], [-1, -1], [1, -1]]
    X = np.zeros((n_samples, n_features))
    y = np.zeros(n_samples, dtype=int)

    for i in range(3):
        start_idx = i * 100
        end_idx = (i + 1) * 100
        X[start_idx:end_idx, :2] = np.random.randn(100, 2) * 0.4 + centers[i]
        y[start_idx:end_idx] = i

    # 2. Run Pillar 1: Scaffolding
    laplacian = scaffolding.construct_laplacian(X, k=10)
    print(f"Laplacian created with shape: {laplacian.shape}")

    # 3. Run Pillar 2: Decomposition
    eigenvalues, eigenvectors = decomposition.spectral_decomposition(laplacian, m=10)
    print(f"Eigenvectors computed with shape: {eigenvectors.shape}")

    # 4. Perform clustering on eigenvectors (a simple use case)
    print("Performing K-Means on the first few non-trivial eigenvectors...")
    # We skip the first eigenvector (v0), which is constant for connected components
    spectral_embedding = eigenvectors[:, 1:4]
    kmeans = KMeans(n_clusters=3, random_state=42, n_init='auto')
    clusters = kmeans.fit_predict(spectral_embedding)
    print(f"Discovered clusters: {np.unique(clusters, return_counts=True)}")

    # 5. Call placeholder for Pillar 3
    analysis.compute_geodesic_distance()

    print("\n--- Principia Semantica Pipeline Test Complete ---")


if __name__ == "__main__":
    main()