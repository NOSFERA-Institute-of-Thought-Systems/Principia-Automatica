# paper-01-spectral-geometry/scripts/paper1_analyze.py

import numpy as np
import faiss
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigs
import time
import os
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans

def c_tf_idf(documents_per_cluster, n_words=10):
    """
    Calculates Class-based TF-IDF scores.
    This version is robust to empty clusters and documents becoming empty after stopword removal.
    
    documents_per_cluster: A list of lists of strings. 
                           Each inner list contains the documents for one cluster.
    """
    # 1. Pre-process and create a vocabulary for all documents
    all_docs = [doc for cluster in documents_per_cluster for doc in cluster]
    if not all_docs:
        return {i: ["EMPTY UNIVERSE"] * n_words for i in range(len(documents_per_cluster))}

    count_vectorizer = CountVectorizer(stop_words="english").fit(all_docs)
    vocab = count_vectorizer.vocabulary_
    feature_names = count_vectorizer.get_feature_names_out()

    # 2. Calculate term frequencies (TF) for each cluster
    tfs = []
    for docs in documents_per_cluster:
        if not docs:
            tfs.append(csr_matrix((1, len(vocab)), dtype=int))
            continue
        
        # Use the pre-built vocabulary
        cv_cluster = CountVectorizer(vocabulary=vocab)
        t = cv_cluster.fit_transform(docs)
        tfs.append(t.sum(axis=0))

    tf_matrix = csr_matrix(np.vstack([tf for tf in tfs]))

    # 3. Calculate Inverse Document Frequency (IDF)
    # Total number of words per cluster
    w = np.asarray(tf_matrix.sum(axis=1)).ravel()
    
    # Total number of words across all clusters
    total_words = tf_matrix.sum()
    
    # Term frequencies across all clusters
    sum_t = np.asarray(tf_matrix.sum(axis=0)).ravel()
    
    # IDF formula
    idf = np.log(total_words / (sum_t + 1e-9)) # Add epsilon to avoid log(0)

    # 4. Calculate TF-IDF
    # Normalize TF by the word count of each cluster
    w[w == 0] = 1e-9 # Avoid division by zero
    tf_normalized = tf_matrix.multiply(1 / w[:, np.newaxis])
    
    tf_idf_matrix = tf_normalized.multiply(idf)
    tf_idf_matrix = np.asarray(tf_idf_matrix.todense())

    # 5. Extract top words
    top_words = {}
    for i in range(len(documents_per_cluster)):
        if w[i] < 1: # Check if cluster was effectively empty
            top_words[i] = ["EMPTY CLUSTER"] * n_words
        else:
            indices = np.argsort(tf_idf_matrix[i, :])[-n_words:]
            top_words[i] = [feature_names[j] for j in indices[::-1]]
            
    return top_words


def main():
    print("--- Starting Full Paper 1 Analysis Pipeline ---")
    
    # ... Loading and setup code remains the same ...
    data_path = "paper-01-spectral-geometry/data/embeddings.npy"
    print(f"Loading embeddings from {data_path}...")
    embeddings = np.load(data_path).astype('float32')
    n_docs, dim = embeddings.shape
    print(f"-> Loaded {n_docs} embeddings of dimension {dim}.")
    newsgroups_data = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
    documents_text = newsgroups_data.data

    k = 10
    print(f"\nConstructing k-NN graph with k={k} using Faiss...")
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    distances, indices = index.search(embeddings, k + 1)
    
    print("\nBuilding Adjacency and Laplacian matrices...")
    rows = np.arange(n_docs).repeat(k)
    cols = indices[:, 1:].flatten()
    adjacency = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n_docs, n_docs))
    adjacency = adjacency + adjacency.T
    degree = np.array(adjacency.sum(axis=1)).flatten()
    degree_inv_sqrt = 1.0 / np.sqrt(degree)
    degree_inv_sqrt[np.isinf(degree_inv_sqrt)] = 0
    D_inv_sqrt = csr_matrix(np.diag(degree_inv_sqrt))
    laplacian = csr_matrix(np.eye(n_docs)) - D_inv_sqrt @ adjacency @ D_inv_sqrt

    num_eigenvectors = 20
    print(f"\nComputing the smallest {num_eigenvectors} eigenvectors...")
    eigenvalues, eigenvectors = eigs(laplacian, k=num_eigenvectors, which='SM')
    order = np.argsort(np.real(eigenvalues))
    eigenvectors = np.real(eigenvectors[:, order])
    output_dir = "paper-01-spectral-geometry/results"
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, 'spectral_embedding.npy'), eigenvectors)
    
    embedding_for_clustering = eigenvectors[:, 1:16]
    n_clusters = 20
    print(f"\nPerforming K-Means clustering with {n_clusters} clusters...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    discovered_labels = kmeans.fit_predict(embedding_for_clustering)
    np.save(os.path.join(output_dir, 'discovered_labels.npy'), discovered_labels)

    print("\nSummarizing discovered topics using c-TF-IDF...")
    docs_by_cluster = [[] for _ in range(n_clusters)]
    for i, label in enumerate(discovered_labels):
        docs_by_cluster[label].append(documents_text[i])
    
    # Pass the documents grouped by cluster to the robust function
    topics = c_tf_idf(docs_by_cluster, n_words=10)
    
    print("\n--- Discovered Topics (Top 10 words) ---")
    with open(os.path.join(output_dir, 'discovered_topics.txt'), 'w') as f:
        for i in sorted(topics.keys()):
            topic_str = f"Topic {i}: {' '.join(topics[i])}"
            print(topic_str)
            f.write(topic_str + '\n')
            
    print(f"\nFull topic list saved to '{output_dir}/discovered_topics.txt'")
    print("\n--- Full Analysis Pipeline Complete ---")


if __name__ == "__main__":
    main()