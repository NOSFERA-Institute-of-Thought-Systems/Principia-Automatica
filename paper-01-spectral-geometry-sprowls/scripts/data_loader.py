# paper-01-spectral-geometry/scripts/data_loader.py

import arxiv
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from typing import List, Tuple, Optional
import os
import numpy as np

def load_20newsgroups() -> Tuple[List[str], Optional[List[int]]]:
    """
    Loads the 20 Newsgroups dataset.

    Returns:
        A tuple containing:
        - A list of document texts.
        - A list of integer ground-truth labels.
    """
    print("Fetching 20 Newsgroups dataset...")
    dataset = fetch_20newsgroups(
        subset='all', 
        remove=('headers', 'footers', 'quotes')
    )
    print(f"-> Found {len(dataset.data)} documents.")
    return dataset.data, dataset.target

def load_arxiv_abstracts(query="cat:cs.CL", max_results=1000) -> Tuple[List[str], Optional[List[int]]]:
    """
    Loads computer science abstracts from the ArXiv API.
    `cs.CL` = Computational Linguistics. A thematically coherent set.

    Args:
        query: The ArXiv query string.
        max_results: The maximum number of abstracts to download.

    Returns:
        A tuple containing:
        - A list of document texts (title + summary).
        - None for labels, as this is unsupervised.
    """
    print(f"Querying ArXiv for '{query}' with max {max_results} results...")
    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate
    )
    
    documents = []
    for result in client.results(search):
        # --- THIS IS THE CRITICAL FIX ---
        # The abstract text is stored in the 'summary' attribute.
        documents.append(f"{result.title}. {result.summary}")
        # --- END OF FIX ---
        
    print(f"-> Found {len(documents)} documents.")
    return documents, None

def load_reddit_comments(filepath: str, subreddit: str, min_length: int = 50) -> Tuple[List[str], Optional[List[int]]]:
    """
    Loads Reddit comments from a pre-downloaded CSV file.
    Assumes the CSV has 'subreddit' and 'body' columns.
    """
    print(f"Loading Reddit comments for 'r/{subreddit}' from {filepath}...")
    try:
        # Specify dtype for 'body' to prevent type inference issues
        df = pd.read_csv(filepath, dtype={'body': str, 'subreddit': str})
        # Filter for the specific subreddit
        df_sub = df[df['subreddit'] == subreddit]
        # Filter for comments that are not deleted and have a minimum length
        # .str accessor requires non-null values
        df_filtered = df_sub[
            df_sub['body'].notna() &
            (df_sub['body'] != '[deleted]') & 
            (df_sub['body'] != '[removed]') & 
            (df_sub['body'].str.len() >= min_length)
        ]
        documents = df_filtered['body'].tolist()
        print(f"-> Found {len(documents)} valid comments.")
        return documents, None
    except FileNotFoundError:
        print(f"-> ERROR: File not found at {filepath}.")
        print("-> Please download a Reddit dataset and place it there.")
        return [], None
    except Exception as e:
        print(f"-> An error occurred while processing the CSV: {e}")
        return [], None

if __name__ == '__main__':
    # This block allows you to test the functions directly
    print("--- Testing Data Loaders ---")
    
    docs_20ng, _ = load_20newsgroups()
    print("-" * 20)
    
    docs_arxiv, _ = load_arxiv_abstracts(max_results=50) # Small query for testing
    print("-" * 20)
    
    # Create a dummy CSV for testing the Reddit loader
    dummy_data = {
        'subreddit': ['askscience', 'science', 'askscience', 'askscience', None],
        'body': [
            'This is a long and detailed comment about quantum physics that meets the length requirement.',
            'This is a comment from the wrong subreddit.',
            '[deleted]',
            'Too short.',
            np.nan # Add a null value to test robustness
        ]
    }
    dummy_df = pd.DataFrame(dummy_data)
    dummy_filepath = "data/dummy_reddit.csv"
    os.makedirs("data", exist_ok=True)
    dummy_df.to_csv(dummy_filepath, index=False)
    
    docs_reddit, _ = load_reddit_comments(dummy_filepath, 'askscience')
    print("Test Reddit comment:", docs_reddit[0] if docs_reddit else "None")