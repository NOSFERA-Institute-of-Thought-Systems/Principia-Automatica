# experiments/run_umap_wavelet_experiment.py
import sys
import os
import argparse
import numpy as np
import umap
import networkx as nx
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data import data_loader
from principia_semantica import scaffolding, decomposition, utils, wavelets, visualizer, curvature

@utils.timeit
def run_final_experiment(num_scales: int, umap_dims: int, umap_neighbors: int):
    print(f"\n--- Running Final Experiment: UMAP-Wavelet + Curvature Analysis ---")
    
    embeddings, true_labels = data_loader.load_and_embed_20newsgroups()
    
    # We need the graph object for curvature, so we modify the scaffolding call
    laplacian, G = scaffolding.construct_laplacian(embeddings, k=20, return_graph=True)
    eigenvalues, eigenvectors = decomposition.spectral_decomposition(laplacian, m=20)
    
    wavelet_features = wavelets.compute_wavelet_features(
        eigenvalues, eigenvectors, num_scales=num_scales
    )
    
    reducer = umap.UMAP(
        n_components=umap_dims, n_neighbors=umap_neighbors, 
        metric='cosine', random_state=42
    )
    reduced_features = reducer.fit_transform(wavelet_features)
    
    kmeans = KMeans(n_clusters=20, random_state=42, n_init='auto')
    final_labels = kmeans.fit_predict(reduced_features)
    ari_score = adjusted_rand_score(true_labels, final_labels)
    
    print(f"--> Final UMAP-Wavelet ARI Score: {ari_score:.4f}")

    # --- Pillar 3: Curvature Analysis ---
    G_with_curvature = curvature.compute_ricci_curvature(G)
    
    # Calculate node curvature as the average of its edge curvatures
    node_curvatures = np.zeros(G.number_of_nodes())
    for i in range(G.number_of_nodes()):
        if G.degree(i) > 0:
            node_curvatures[i] = np.mean([d['ricciCurvature'] for _, _, d in G_with_curvature.edges(i, data=True)])
            
    # --- Final Visualizations ---
    results_dir = "results/20_newsgroups_final"
    visualizer.visualize_clusters(
        embeddings, final_labels, 
        title=f"Final Discovered Clusters (UMAP-Wavelet) - ARI {ari_score:.3f}",
        output_dir=results_dir
    )
    visualizer.visualize_clusters(
        embeddings, node_curvatures, 
        title="Conceptual Space Colored by Ricci Curvature",
        output_dir=results_dir,
        is_continuous=True # Tell visualizer to use a continuous colormap
    )
    print("\n--- Final Experiment and Analysis Complete ---")

# (main function remains the same, just calls run_final_experiment)
def main():
    parser = argparse.ArgumentParser(description="Run the final UMAP+Curvature experiment.")
    parser.add_argument('--scales', type=int, default=10, help="Number of wavelet scales.")
    parser.add_argument('--umap_dims', type=int, default=10, help="Number of UMAP components.")
    parser.add_argument('--umap_neighbors', type=int, default=15, help="Number of UMAP neighbors.")
    args = parser.parse_args()

    run_final_experiment(
        num_scales=args.scales, 
        umap_dims=args.umap_dims, 
        umap_neighbors=args.umap_neighbors
    )

if __name__ == "__main__":
    main()