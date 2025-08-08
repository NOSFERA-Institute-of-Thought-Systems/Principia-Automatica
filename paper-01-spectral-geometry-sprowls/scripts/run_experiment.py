# paper-01-spectral-geometry-sprowls/scripts/run_experiment.py
import os
import argparse
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import data_loader
import spectral_analyzer
import visualizer
from benchmarker import run_lda, run_bertopic
from spectral_analyzer import c_tf_idf

def save_topics(topics, filepath):
    if not topics: return
    with open(filepath, 'w') as f:
        for i in sorted(topics.keys()):
            f.write(f"Topic {i}: {' '.join(topics[i])}\n")

def main():
    parser = argparse.ArgumentParser(description="Run Spectral Geometry experiments for Paper 1.")
    parser.add_argument('--dataset', type=str, required=True, choices=['20ng', 'arxiv', 'reddit'], help="The dataset to process.")
    parser.add_argument('--k', type=int, default=15, help="Number of nearest neighbors.")
    parser.add_argument('--m', type=int, default=20, help="Number of eigenvectors.")
    parser.add_argument('--n_clusters', type=int, default=20, help="Number of clusters to find.")
    args = parser.parse_args()

    print(f"--- Running Full Experiment on '{args.dataset}' Dataset ---")
    
    # --- THIS IS THE CRITICAL FIX ---
    # 1. Load Data using a mapping from short name to full function name
    data_loader_map = {
        '20ng': data_loader.load_20newsgroups,
        'arxiv': data_loader.load_arxiv_abstracts,
        'reddit': data_loader.load_reddit_comments,
    }
    
    if args.dataset not in data_loader_map:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    # For reddit, we need to pass arguments, for others we don't.
    if args.dataset == 'reddit':
        # NOTE: Update this path to your REAL Reddit data when you have it.
        filepath = "paper-01-spectral-geometry-sprowls/data/dummy_reddit.csv"
        documents, true_labels = data_loader_map[args.dataset](filepath, 'askscience')
    else:
        documents, true_labels = data_loader_map[args.dataset]()
    # --- END OF FIX ---
    
    if not documents:
        print("No documents found for this dataset. Exiting.")
        return

    # 2. Embedding
    output_dir = f"paper-01-spectral-geometry-sprowls/data/{args.dataset}"
    os.makedirs(output_dir, exist_ok=True)
    embeddings_path = os.path.join(output_dir, 'embeddings.npy')
    if os.path.exists(embeddings_path):
        print(f"Loading existing embeddings from {embeddings_path}...")
        embeddings = np.load(embeddings_path)
    else:
        print("Generating embeddings...")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = model.encode(documents, show_progress_bar=True)
        np.save(embeddings_path, embeddings)

    # 3. Spectral Analysis
    eigenvalues, eigenvectors = spectral_analyzer.analyze(embeddings, k=args.k, num_eigenvectors=args.m)
    
    # 4. Clustering
    embedding_for_clustering = eigenvectors[:, 1:args.m]
    print(f"\nPerforming K-Means clustering with {args.n_clusters} clusters...")
    kmeans = KMeans(n_clusters=args.n_clusters, random_state=42, n_init='auto')
    discovered_labels = kmeans.fit_predict(embedding_for_clustering)

    # 5. Summarization
    print("\nSummarizing HSM topics...")
    docs_by_cluster = [[] for _ in range(args.n_clusters)]
    for i, label in enumerate(discovered_labels): docs_by_cluster[label].append(documents[i])
    hsm_topics = c_tf_idf(docs_by_cluster, n_words=10)

    # 6. Run Baselines
    lda_topics = run_lda(documents, n_topics=args.n_clusters)
    bertopic_topics = run_bertopic(documents)

    # 7. Save Results
    results_dir = f"paper-01-spectral-geometry-sprowls/results/{args.dataset}"
    os.makedirs(results_dir, exist_ok=True)
    save_topics(hsm_topics, os.path.join(results_dir, 'hsm_topics.txt'))
    save_topics(lda_topics, os.path.join(results_dir, 'lda_topics.txt'))
    save_topics(bertopic_topics, os.path.join(results_dir, 'bertopic_topics.txt'))
    print(f"\nAll topic models have been run. Results saved in {results_dir}")

    # 8. Visualization
    if true_labels is not None:
        visualizer.visualize_clusters(
            embeddings, true_labels, f"UMAP of {args.dataset} (Ground Truth)",
            os.path.join(results_dir, 'umap_ground_truth.png')
        )
    visualizer.visualize_clusters(
        embeddings, discovered_labels, f"UMAP of {args.dataset} (HSM Discovered)",
        os.path.join(results_dir, 'umap_hsm_discovered.png')
    )
    
    print("\n--- Full Experiment Complete ---")

if __name__ == '__main__':
    main()