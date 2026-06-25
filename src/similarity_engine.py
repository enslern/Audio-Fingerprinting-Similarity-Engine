"""
similarity_engine.py
Computes cosine similarity between song embeddings and returns top matches.
"""

import numpy as np


def cosine_similarity(vec_a, vec_b):
    a = np.array(vec_a)
    b = np.array(vec_b)
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def find_similar(query_song, features_db, top_n=3):
    """
    query_song: filename key in features_db
    features_db: dict loaded from data/features.json
    Returns list of (song_name, similarity_score) sorted descending.
    """
    if query_song not in features_db:
        raise ValueError(f"'{query_song}' not found in features database.")

    query_embedding = features_db[query_song]["embedding"]
    scores = []

    for song, data in features_db.items():
        if song == query_song:
            continue
        sim = cosine_similarity(query_embedding, data["embedding"])
        scores.append((song, sim))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_n]
