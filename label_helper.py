import argparse
import json
import os

from search import HybridSearcher

ap = argparse.ArgumentParser()
ap.add_argument("query")
ap.add_argument("near")
ap.add_argument("radius_m", type=float)
ap.add_argument("--keywords", nargs="*", default=[],
                help="words to look for in name/category/tags/description, e.g. vegan books indian")
ap.add_argument("--top", type=int, default=15, help="how many semantic results to add to the pool")
args = ap.parse_args()

s = HybridSearcher()
anchor = s.store.geocode(args.near)
if anchor is None:
    raise SystemExit(f"Could not find '{args.near}'")
print(f"Anchor: {anchor['name']} ({anchor['category']})")

hits = s.store.within_radius(anchor["lat"], anchor["lon"], args.radius_m)
print(f"{len(hits)} places within {args.radius_m:.0f} m")


def text_of(p):
    return " ".join([p["name"], p["category"], p["description"],
                     " ".join(f"{k}={v}" for k, v in p["tags"].items())]).lower()


# Source 1: keyword matches in the map data (independent of the embeddings)
kw = [k.lower() for k in args.keywords]
pool = {}
for i, d in hits:
    p = s.store.places[i]
    if kw and any(k in text_of(p) for k in kw):
        pool[p["id"]] = (p, d, "tag/keyword match")

# Source 2: what our semantic search returns
for r in s.filter_then_rank(args.query, anchor["lat"], anchor["lon"], args.radius_m, k=args.top):
    if r["id"] not in pool:
        p = next(x for x in s.store.places if x["id"] == r["id"])
        pool[r["id"]] = (p, r["distance_m"], f"semantic sim {r['similarity']}")

print(f"\n{len(pool)} candidates to judge. Answer y (relevant), n (not), q (stop and save).\n")
relevant = []
for n, (pid, (p, d, why)) in enumerate(sorted(pool.items(), key=lambda x: x[1][1]), 1):
    print(f"[{n}/{len(pool)}] {p['name']}  ({p['category']}, {d:.0f} m)  <- {why}")
    print(f"      {p['description']}")
    ans = input("      relevant? [y/n/q] ").strip().lower()
    if ans == "q":
        break
    if ans == "y":
        relevant.append(pid)

path = "eval_queries.json"
data = []
if os.path.exists(path):
    text = open(path, encoding="utf-8").read().strip()
    if text:  # an empty file just means "no queries yet"
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            os.replace(path, path + ".broken")
            print(f"Note: {path} was not valid JSON; moved it to {path}.broken and started fresh")
data = [x for x in data if not (x["query"] == args.query and x["near"] == args.near)]
data.append({"query": args.query, "near": args.near, "radius_m": args.radius_m,
             "relevant_ids": relevant})
json.dump(data, open(path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
print(f"\nSaved {len(relevant)} relevant places for '{args.query}' to {path}")
