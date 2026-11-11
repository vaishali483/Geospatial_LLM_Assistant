"""Step 3b: hybrid search = geography first, then meaning.

Try:  python search.py "quiet cafe with wifi" "Birmingham New Street" 500
"""
import sys

import faiss
import numpy as np

from geo import PlaceStore, haversine_m, load_places
from step3_build_index import embed, get_model


class HybridSearcher:
    def __init__(self):
        self.store = PlaceStore(load_places())
        self.vecs = np.load("data/embeddings.npy")
        self.flat = faiss.read_index("data/flat.index")
        self.model = get_model()

    def embed_query(self, text):
        return embed(self.model, [text])  # shape (1, dim)

    def filter_then_rank(self, query, lat, lon, radius_m, category=None,
                         step_free_only=False, k=5):
        """Strategy A (our main one): exact radius filter, then rank survivors by meaning."""
        hits = self.store.within_radius(lat, lon, radius_m)
        results = []
        for i, dist in hits:
            p = self.store.places[i]
            if category and p["category"] != category:
                continue
            if step_free_only and p["step_free"] != "yes":
                continue
            results.append((i, dist))
        if not results:
            return []
        q = self.embed_query(query)[0]
        idx = np.array([i for i, _ in results])
        sims = self.vecs[idx] @ q  # cosine similarity, because vectors are normalised
        order = np.argsort(-sims)[:k]
        return [self._row(results[j][0], results[j][1], float(sims[j])) for j in order]

    def search_then_filter(self, query, lat, lon, radius_m, k=5, fetch=100):
        """Strategy B (for comparison): global vector search, then drop far-away places.
        Fast with a big index, but can return nothing if all top hits are elsewhere."""
        q = self.embed_query(query)
        sims, ids = self.flat.search(q, fetch)
        out = []
        for i, s in zip(ids[0], sims[0]):
            if i < 0:
                continue
            p = self.store.places[i]
            d = float(haversine_m(lat, lon, p["lat"], p["lon"]))
            if d <= radius_m:
                out.append(self._row(int(i), d, float(s)))
            if len(out) == k:
                break
        return out

    def _row(self, i, dist, sim):
        p = self.store.places[i]
        return {"id": p["id"], "name": p["name"], "category": p["category"],
                "distance_m": round(dist), "similarity": round(sim, 3),
                "step_free": p["step_free"], "lat": p["lat"], "lon": p["lon"],
                "description": p["description"]}


def show(title, rows):
    print(f"\n{title}")
    if not rows:
        print("  (nothing found)")
    for r in rows:
        print(f"  {r['distance_m']:5d} m  sim {r['similarity']:.3f}  {r['name']:<32} "
              f"step-free: {r['step_free']}")


if __name__ == "__main__":
    query = sys.argv[1] if len(sys.argv) > 1 else "quiet cafe with wifi"
    place = sys.argv[2] if len(sys.argv) > 2 else "Birmingham New Street"
    radius = float(sys.argv[3]) if len(sys.argv) > 3 else 500

    s = HybridSearcher()
    anchor = s.store.geocode(place)
    if anchor is None:
        raise SystemExit(f"Could not find '{place}' in the data")
    print(f"Query: '{query}' within {radius:.0f} m of {anchor['name']}")

    show("A) Filter by distance, then rank by meaning:",
         s.filter_then_rank(query, anchor["lat"], anchor["lon"], radius))
    show("A) ...cafes only, confirmed step-free only:",
         s.filter_then_rank(query, anchor["lat"], anchor["lon"], radius,
                            category="cafe", step_free_only=True))
    show("B) Search by meaning across the whole region, then filter by distance:",
         s.search_then_filter(query, anchor["lat"], anchor["lon"], radius))
