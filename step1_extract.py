"""Step 1: read the OSM file and turn useful places into clean records.

Run:  python step1_extract.py data/west-midlands-261108.osm.pbf
Out:  data/places.jsonl  (one JSON record per line)
"""
import json
import sys
from pathlib import Path

import osmium

# Which map features we keep. Key = OSM tag key, value = allowed values (None = any value).
KEEP = {
    "amenity": {"cafe", "restaurant", "pub", "bar", "fast_food", "library",
                "pharmacy", "cinema", "theatre", "arts_centre", "community_centre"},
    "shop": None,
    "tourism": {"museum", "gallery", "attraction", "hotel"},
    "leisure": {"park", "sports_centre", "fitness_centre"},
    "railway": {"station"},
}

# Tags worth carrying into the record (everything else is dropped).
USEFUL_TAGS = ["cuisine", "opening_hours", "wheelchair", "outdoor_seating",
               "internet_access", "diet:vegan", "diet:vegetarian", "takeaway",
               "website", "description", "addr:street", "addr:postcode"]


def category_of(tags):
    """Return e.g. ('amenity', 'cafe') if the object is something we keep, else None."""
    for key, allowed in KEEP.items():
        value = tags.get(key)
        if value and (allowed is None or value in allowed):
            return key, value
    return None


def step_free_of(tags):
    """Three-valued accessibility: missing tag means UNKNOWN, never 'no'."""
    w = tags.get("wheelchair")
    if w in ("yes", "designated"):
        return "yes"
    if w in ("limited", "no"):
        return w
    return "unknown"


def describe(name, key, value, tags):
    """Build the text we will embed, from tags only (so it cannot invent anything).
    Coordinates are deliberately NOT included."""
    kind = value.replace("_", " ")
    if key == "shop":
        kind = f"{kind} shop"
    parts = [f"{name} is a {kind}."]
    if tags.get("cuisine"):
        parts.append("Food: " + tags["cuisine"].replace(";", ", ").replace("_", " ") + ".")
    if tags.get("diet:vegan") in ("yes", "only"):
        parts.append("Vegan options.")
    if tags.get("diet:vegetarian") in ("yes", "only"):
        parts.append("Vegetarian options.")
    if tags.get("outdoor_seating") == "yes":
        parts.append("Has outdoor seating.")
    if tags.get("internet_access") in ("wlan", "yes", "wifi"):
        parts.append("Has wifi.")
    if tags.get("takeaway") in ("yes", "only"):
        parts.append("Offers takeaway.")
    if tags.get("description"):
        parts.append(tags["description"])
    return " ".join(parts)


def make_record(osm_type, osm_id, tags, lat, lon):
    tags = dict(tags)
    name = tags.get("name")
    cat = category_of(tags)
    if not name or not cat:
        return None
    key, value = cat
    return {
        "id": f"osm:{osm_type}/{osm_id}",
        "name": name,
        "category": value,
        "category_key": key,
        "lat": round(lat, 7),
        "lon": round(lon, 7),
        "step_free": step_free_of(tags),
        "tags": {t: tags[t] for t in USEFUL_TAGS if t in tags},
        "description": describe(name, key, value, tags),
    }


def main(pbf_path, out_path="data/places.jsonl"):
    records = []
    # with_locations() lets us look up the coordinates of the nodes that make up a way.
    for obj in osmium.FileProcessor(pbf_path).with_locations():
        if not obj.tags or "name" not in obj.tags:
            continue
        if obj.is_node():
            if not obj.location.valid():
                continue
            rec = make_record("node", obj.id, obj.tags, obj.location.lat, obj.location.lon)
        elif obj.is_way():
            # A building or area: use the average of its corner points as one representative point.
            pts = [(n.lat, n.lon) for n in obj.nodes if n.location.valid()]
            if not pts:
                continue
            lat = sum(p[0] for p in pts) / len(pts)
            lon = sum(p[1] for p in pts) / len(pts)
            rec = make_record("way", obj.id, obj.tags, lat, lon)
        else:
            continue  # relations skipped for the MVP (see guide)
        if rec:
            records.append(rec)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Saved {len(records):,} places to {out_path}")
    counts = {}
    for r in records:
        counts[r["category"]] = counts.get(r["category"], 0) + 1
    for cat, n in sorted(counts.items(), key=lambda x: -x[1])[:10]:
        print(f"  {cat:<20} {n:,}")
    sf = sum(r["step_free"] == "unknown" for r in records)
    print(f"Accessibility unknown for {sf:,} of {len(records):,} places")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/west-midlands-261108.osm.pbf")
