# Utils.py

import pandas as pd
from transformers import BertTokenizer, BertModel
import torch
from sklearn.neighbors import NearestNeighbors
import numpy as np

def load_data(file_path):
    """
    Load the data from a CSV file.
    """
    return pd.read_csv(file_path, usecols=['Title', 'Abstract'])

def create_embeddings(data):
    """
    Create embeddings for the data using BERT.
    """
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')

    embeddings = []
    for _, row in data.iterrows():
        inputs = tokenizer(row['Title'] + " " + row['Abstract'], return_tensors="pt", truncation=True, max_length=512)
        outputs = model(**inputs)
        embeddings.append(outputs.last_hidden_state.mean(1).detach().numpy())
    
    return np.vstack(embeddings)

def find_nearest_neighbors(embeddings, query_embedding, query_index=None, n_neighbors=5):
    """
    Find the nearest neighbors for a query embedding.
    """
    nn = NearestNeighbors(n_neighbors=n_neighbors + 1, algorithm='ball_tree').fit(embeddings)
    distances, indices = nn.kneighbors([query_embedding])
    
    # Exclude the query index itself if provided
    if query_index is not None:
        indices = indices[0][indices[0] != query_index]
    else:
        indices = indices[0]
    
    # Return the top n_neighbors indices, excluding the query index
    return indices[:n_neighbors]