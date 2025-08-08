# paper-01-spectral-geometry-sprowls/scripts/spectral_analyzer.py
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import csr_matrix, eye as sparse_eye
from scipy.sparse.linalg import eigs
import time

def c_tf_idf(documents_per_cluster, n_words=10):
    all_docs = [doc for cluster in documents_per_cluster for doc in cluster]
    if not all_docs: return {i: ["EMPTY UNIVERSE"]*n_words for i in range(len(documents_per_cluster))}
    count_vectorizer = CountVectorizer(stop_words="english").fit(all_docs)
    vocab = count_vectorizer.vocabulary_
    feature_names = count_vectorizer.get_feature_names_out()
    tfs = [CountVectorizer(vocabulary=vocab).fit_transform(docs).sum(axis=0) if docs else csr_matrix((1, len(vocab))) for docs in documents_per_cluster]
    tf_matrix = csr_matrix(np.vstack(tfs))
    w = np.asarray(tf_matrix.sum(axis=1)).ravel()
    sum_t = np.asarray(tf_matrix.sum(axis=0)).ravel()
    idf = np.log(tf_matrix.sum() / (sum_t + 1e-9))
    w[w == 0] = 1e-9
    tf_normalized = tf_matrix.multiply(1 / w[:, np.newaxis])
    tf_idf_matrix = np.asarray(tf_normalized.multiply(idf).todense())
    top_words = {}
    for i in range(len(documents_per_cluster)):
        if w[i] < 1: top_words[i] = ["EMPTY CLUSTER"]*n_words
        else: top_words[i] = [feature_names[j] for j in np.argsort(tf_idf_matrix[i, :])[-n_words:][::-1]]
    return top_words

def analyze(embeddings: np.ndarray, k: int = 15, num_eigenvectors: int = 20) -> tuple[np.ndarray, np.ndarray]:
    n_docs, _ = embeddings.shape
    print(f"\n--- Starting Spectral Analysis on {n_docs} documents ---")
    embeddings = embeddings.astype('float32')
    print(f"Constructing k-NN graph with k={k} using scikit-learn...")
    nn = NearestNeighbors(n_neighbors=k + 1, metric='cosine', n_jobs=-1)
    nn.fit(embeddings)
    start_time = time.time()
    _, indices = nn.kneighbors(embeddings)
    print(f"-> k-NN search complete in {time.time() - start_time:.2f}s.")
    print("Building Adjacency and Laplacian matrices...")
    rows, cols = np.arange(n_docs).repeat(k), indices[:, 1:].flatten()
    adjacency = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n_docs, n_docs))
    adjacency = adjacency + adjacency.T
    degree = np.array(adjacency.sum(axis=1)).flatten()
    degree_inv_sqrt = 1.0 / np.sqrt(degree, where=(degree > 0))
    D_inv_sqrt = csr_matrix(np.diag(degree_inv_sqrt))
    laplacian = sparse_eye(n_docs, format='csr') - D_inv_sqrt @ adjacency @ D_inv_sqrt
    print("-> Laplacian built.")
    print(f"Computing the smallest {num_eigenvectors} eigenvectors...")
    start_time = time.time()
    eigenvalues, eigenvectors = eigs(laplacian, k=num_eigenvectors, which='SM')
    print(f"-> Eigendecomposition complete in {time.time() - start_time:.2f}s.")
    order = np.argsort(np.real(eigenvalues))
    eigenvalues, eigenvectors = np.real(eigenvalues[order]), np.real(eigenvectors[:, order])
    print("Smallest Eigenvalues:", eigenvalues[:5])
    return eigenvalues, eigenvectors