"""Step 2: geography done with plain maths, never by the LLM."""
import json

import numpy as np

EARTH_RADIUS_M = 6_371_000


def load_places(path="data/places.jsonl"):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def haversine_m(lat1, lon1, lat2, lon2):
    """Great-circle distance in metres. Works on single numbers or numpy arrays."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(a))


class PlaceStore:
    """Holds all places plus their coordinates as numpy arrays for fast maths."""

    def __init__(self, places):
        self.places = places
        self.lats = np.array([p["lat"] for p in places])
        self.lons = np.array([p["lon"] for p in places])

    def geocode(self, name):
        """Find a place's coordinates by name, using OUR data (no external service).
        Prefers an exact name match, then a station, then any partial match."""
        q = name.lower().strip()
        exact = [p for p in self.places if p["name"].lower() == q]
        partial = [p for p in self.places if q in p["name"].lower()]
        for group in (exact, [p for p in partial if p["category"] == "station"], partial):
            if group:
                return group[0]
        return None

    def within_radius(self, lat, lon, radius_m):
        """Return (index, distance_m) for every place within radius_m, nearest first."""
        # 1) cheap bounding-box prefilter: 1 degree of latitude is about 111 km,
        #    1 degree of longitude is about 111 km * cos(latitude).
        dlat = radius_m / 111_000
        dlon = radius_m / (111_000 * np.cos(np.radians(lat)))
        box = np.where((np.abs(self.lats - lat) <= dlat) & (np.abs(self.lons - lon) <= dlon))[0]
        # 2) exact distance only for the few places inside the box
        d = haversine_m(lat, lon, self.lats[box], self.lons[box])
        keep = d <= radius_m
        idx, dist = box[keep], d[keep]
        order = np.argsort(dist)
        return list(zip(idx[order].tolist(), dist[order].tolist()))


if __name__ == "__main__":
    store = PlaceStore(load_places())
    print(f"Loaded {len(store.places):,} places")

    anchor = store.geocode("Birmingham New Street")
    if anchor is None:
        raise SystemExit("Could not find 'Birmingham New Street' in the data")
    print(f"Anchor: {anchor['name']} ({anchor['category']}) at {anchor['lat']}, {anchor['lon']}")

    hits = store.within_radius(anchor["lat"], anchor["lon"], 500)
    cafes = [(i, d) for i, d in hits if store.places[i]["category"] == "cafe"]
    print(f"{len(hits)} places within 500 m, of which {len(cafes)} are cafes. Nearest cafes:")
    for i, d in cafes[:10]:
        p = store.places[i]
        print(f"  {d:6.0f} m  {p['name']:<35} step-free: {p['step_free']}")
