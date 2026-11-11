import json
import os
import random
import time

import faiss
import numpy as np

from geo import haversine_m
from search import HybridSearcher

random.seed(0)
s = HybridSearcher()
places, store = s.store.places, s.store
n = len(places)
print(f"{n:,} places loaded\n")

# 1) Geographic correctness: our fast radius search must match brute force exactly.
mismatches = 0
for _ in range(200):
    a = random.choice(places)
    r = random.choice([100, 250, 500, 1000, 2000])
    fast = {i for i, _ in store.within_radius(a["lat"], a["lon"], r)}
    brute = set(np.where(haversine_m(a["lat"], a["lon"], store.lats, store.lons) <= r)[0].tolist())
    mismatches += fast != brute
print(f"1) Radius search vs brute force: {200 - mismatches}/200 anchors identical")

# 2) Approximate (HNSW) vs exact (Flat) search: recall@10 and speed.
hnsw = faiss.read_index("data/hnsw.index")
q_idx = random.sample(range(n), min(200, n))
queries = s.vecs[q_idx]
k = min(10, n)
t0 = time.perf_counter()
_, exact = s.flat.search(queries, k)
flat_ms = (time.perf_counter() - t0) * 1000 / len(q_idx)
print(f"\n2) Exact search: {flat_ms:.3f} ms per query")
for ef in (16, 64, 128):
    hnsw.hnsw.efSearch = ef
    t0 = time.perf_counter()
    _, approx = hnsw.search(queries, k)
    ms = (time.perf_counter() - t0) * 1000 / len(q_idx)
    recall = np.mean([len(set(a) & set(e)) / k for a, e in zip(approx, exact)])
    print(f"   HNSW efSearch={ef:<4} recall@{k} vs exact = {recall:.3f}   {ms:.3f} ms per query")

# 3) End-to-end latency of one hybrid query (includes embedding the query text).
anchor = store.geocode("Birmingham New Street") or places[0]
times = []
for _ in range(50):
    t0 = time.perf_counter()
    s.filter_then_rank("quiet cafe with wifi", anchor["lat"], anchor["lon"], 500)
    times.append((time.perf_counter() - t0) * 1000)
print(f"\n3) Hybrid query latency: p50 {np.percentile(times, 50):.1f} ms, "
      f"p95 {np.percentile(times, 95):.1f} ms")

# 4) Semantic relevance on YOUR hand-labelled queries (make them with label_helper.py).
path = "eval_queries.json"
text = open(path, encoding="utf-8").read().strip() if os.path.exists(path) else ""
if not text:
    print(f"\n4) Skipped: create {path} with label_helper.py (see guide)")
else:
    labelled = json.loads(text)
    precisions, recalls = [], []
    print(f"\n4) Relevance on {len(labelled)} labelled queries (top 5 results each):")
    for item in labelled:
        a = store.geocode(item["near"])
        rows = s.filter_then_rank(item["query"], a["lat"], a["lon"], item["radius_m"], k=5)
        got = {r["id"] for r in rows}
        rel = set(item["relevant_ids"])
        if not rel:
            # The right answer is "nothing here". Any result is a false result.
            print(f"   no right answer exists, system still returned {len(rows)}  '{item['query']}'")
            continue
        hits = len(got & rel)
        precisions.append(hits / 5)
        recalls.append(hits / len(rel))
        print(f"   precision@5 {precisions[-1]:.2f}  recall@5 {recalls[-1]:.2f}  "
              f"({len(rel)} relevant)  '{item['query']}'")
    if precisions:
        print(f"   Mean precision@5 {np.mean(precisions):.2f}, mean recall@5 {np.mean(recalls):.2f}")
