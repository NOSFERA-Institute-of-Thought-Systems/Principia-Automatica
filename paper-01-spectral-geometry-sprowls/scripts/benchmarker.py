# paper-01-spectral-geometry-sprowls/scripts/benchmarker.py
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
try:
    from bertopic import BERTopic
    BERTOPIC_INSTALLED = True
except ImportError:
    BERTOPIC_INSTALLED = False

def get_topics_sklearn(model, feature_names, n_words):
    topics = {}
    for topic_idx, topic in enumerate(model.components_):
        top_features_ind = topic.argsort()[: -n_words - 1 : -1]
        topic_words = [feature_names[i] for i in top_features_ind]
        topics[topic_idx] = topic_words
    return topics

def run_lda(documents, n_topics=20, n_words=10):
    print("\n--- Running LDA Baseline ---")
    vectorizer = CountVectorizer(stop_words='english', max_df=0.9, min_df=5)
    X = vectorizer.fit_transform(documents)
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42, n_jobs=-1)
    lda.fit(X)
    feature_names = vectorizer.get_feature_names_out()
    topics = get_topics_sklearn(lda, feature_names, n_words)
    print("-> LDA complete.")
    return topics

def run_bertopic(documents, n_words=10):
    if not BERTOPIC_INSTALLED:
        print("\n--- Skipping BERTopic (not installed, run 'pip install bertopic') ---")
        return None
    print("\n--- Running BERTopic Baseline ---")
    topic_model = BERTopic(min_topic_size=20, verbose=False)
    topic_model.fit_transform(documents)
    all_topics = topic_model.get_topics()
    topics = {}
    for topic_id, words in all_topics.items():
        if topic_id == -1 or not words: continue
        topics[topic_id] = [word for word, score in words[:n_words]]
    print("-> BERTopic complete.")
    return topics