# paper-01-spectral-geometry/scripts/paper1_preprocess.py

import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sentence_transformers import SentenceTransformer
import os
import time

def main():
    """
    Main function to perform the ETL process:
    1. Fetch the 20 Newsgroups dataset.
    2. Embed the documents into vectors using a pre-trained model.
    3. Save the embeddings to a file.
    """
    print("--- Starting Paper 1 Pre-processing ---")

    # Define the output path for our embeddings
    output_dir = "paper-01-spectral-geometry/data"
    output_path = os.path.join(output_dir, "embeddings.npy")
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # 1. Fetch Data
    print("Step 1/3: Fetching the 20 Newsgroups dataset...")
    # We remove headers, footers, and quotes to get the core text content
    newsgroups_data = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
    documents = newsgroups_data.data
    print(f"-> Fetched {len(documents)} documents.")

    # 2. Embed Documents
    print("\nStep 2/3: Loading Sentence Transformer model (all-MiniLM-L6-v2)...")
    # This model is a good balance of speed and quality, ideal for the M1.
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    print("-> Model loaded. Starting document embedding process...")
    print("(This will take a few minutes on an M1 Mac...)")
    start_time = time.time()
    
    # The encode method processes the documents in batches for efficiency.
    embeddings = model.encode(documents, show_progress_bar=True)
    
    end_time = time.time()
    print(f"-> Embedding complete. Time taken: {end_time - start_time:.2f} seconds.")
    print(f"-> Resulting embedding matrix shape: {embeddings.shape}")

    # 3. Save Embeddings
    print(f"\nStep 3/3: Saving embeddings to '{output_path}'...")
    np.save(output_path, embeddings)
    print("-> Successfully saved.")
    
    print("\n--- Pre-processing complete. ---")
    print(f"Your data is ready for analysis at: {output_path}")


if __name__ == "__main__":
    main()