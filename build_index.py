import time

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from geo import load_places

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # small (384 numbers per text), runs fine on a laptop CPU


def get_model():
    return SentenceTransformer(MODEL_NAME)


def embed(model, texts):
    # normalize_embeddings=True makes every vector length 1,
    # so inner product == cosine similarity.
    vecs = model.encode(texts, batch_size=256, show_progress_bar=True,
                        normalize_embeddings=True)
    return np.asarray(vecs, dtype="float32")


def main():
    places = load_places()
    texts = [p["description"] for p in places]
    print(f"Embedding {len(texts):,} descriptions with {MODEL_NAME} ...")

    t0 = time.perf_counter()
    vecs = embed(get_model(), texts)
    print(f"Done in {time.perf_counter() - t0:.1f} s; shape {vecs.shape}")
    np.save("data/embeddings.npy", vecs)

    dim = vecs.shape[1]

    # Exact search: compares the query with every vector. Our baseline.
    flat = faiss.IndexFlatIP(dim)
    flat.add(vecs)
    faiss.write_index(flat, "data/flat.index")

    # Approximate search: HNSW graph. M = links per node.
    hnsw = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)
    hnsw.hnsw.efConstruction = 200
    t0 = time.perf_counter()
    hnsw.add(vecs)
    print(f"HNSW built in {time.perf_counter() - t0:.1f} s")
    faiss.write_index(hnsw, "data/hnsw.index")

    print("Saved data/embeddings.npy, data/flat.index, data/hnsw.index")


if __name__ == "__main__":
    main()
