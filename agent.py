import os
import sys
from dataclasses import dataclass, field
from typing import Literal

os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")  # hide the start-up banner

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_ai import Agent, ModelRetry, RunContext

from search import HybridSearcher

load_dotenv()  # reads GOOGLE_API_KEY from the .env file

# Check the current Flash model name in Google AI Studio and change it here if needed.
MODEL = "google:gemini-3.8-flash"


# ---------- Typed data the tools return and the agent must answer with ----------

class Location(BaseModel):
    name: str
    category: str
    lat: float
    lon: float


class Place(BaseModel):
    id: str
    name: str
    category: str
    distance_m: int
    similarity: float
    step_free: Literal["yes", "limited", "no", "unknown"]
    description: str
    lat: float
    lon: float


class Answer(BaseModel):
    summary: str = Field(description="A short, friendly answer for the user, in plain words.")
    place_ids: list[str] = Field(description="IDs of the recommended places, copied exactly from tool results.")
    caveats: list[str] = Field(default_factory=list,
                               description="What the data cannot confirm, e.g. unknown step-free access or no data about noise.")


# ---------- Dependencies: things the tools can use (our search engine) ----------

@dataclass
class Deps:
    searcher: HybridSearcher
    seen: dict = field(default_factory=dict)  # every place a tool returned in this run, by ID


agent = Agent(
    MODEL,
    deps_type=Deps,
    output_type=Answer,
    system_prompt=(
        "You help people find places in the West Midlands using OpenStreetMap data. "
        "Always call geocode first to get coordinates, then search_places. "
        "Never guess coordinates, distances, opening hours or accessibility. "
        "Only recommend places returned by search_places, and copy their IDs exactly. "
        "Use the category, cuisine and step_free_only filters for hard requirements; "
        "use the query text for softer wishes. "
        "If step-free access is 'unknown', say so; never treat unknown as yes. "
        "If nothing suitable is found, say that clearly instead of recommending weak matches."
    ),
)


@agent.tool
def geocode(ctx: RunContext[Deps], place_name: str) -> Location:
    """Find the coordinates of a named place (e.g. a station or landmark) in our map data."""
    p = ctx.deps.searcher.store.geocode(place_name)
    if p is None:
        raise ModelRetry(f"'{place_name}' is not in the map data. Try a shorter or different name.")
    return Location(name=p["name"], category=p["category"], lat=p["lat"], lon=p["lon"])


@agent.tool
def search_places(
    ctx: RunContext[Deps],
    query: str,
    lat: float,
    lon: float,
    radius_m: float = 500,
    category: str | None = None,
    cuisine: str | None = None,
    step_free_only: bool = False,
) -> list[Place]:
    """Find places within radius_m metres of (lat, lon), ranked by how well they match the query.

    Args:
        query: what the user wants, in words, e.g. 'quiet cafe with wifi'.
        lat: latitude from geocode.
        lon: longitude from geocode.
        radius_m: search radius in metres.
        category: exact OSM category to require, e.g. 'cafe', 'restaurant', 'library', 'pub', 'books'.
        cuisine: cuisine to require, e.g. 'indian', 'italian', 'vegan'.
        step_free_only: True to return only places confirmed step-free.
    """
    rows = ctx.deps.searcher.filter_then_rank(
        query, lat, lon, radius_m, category=category, cuisine=cuisine,
        step_free_only=step_free_only, k=8,
    )
    places = [Place(**{k: r[k] for k in Place.model_fields}) for r in rows]
    for p in places:
        ctx.deps.seen[p.id] = p
    return places


@agent.output_validator
def only_real_places(ctx: RunContext[Deps], answer: Answer) -> Answer:
    """Grounding check in code: every recommended ID must have come from a tool call."""
    made_up = [pid for pid in answer.place_ids if pid not in ctx.deps.seen]
    if made_up:
        raise ModelRetry(f"These IDs were not returned by search_places: {made_up}. Use only real IDs.")
    return answer


def ask(question: str, searcher: HybridSearcher):
    """Run the agent once. Returns (answer, places, tool_calls)."""
    deps = Deps(searcher=searcher)
    result = agent.run_sync(question, deps=deps)
    answer = result.output
    places = [deps.seen[pid] for pid in answer.place_ids]
    tool_calls = [
        (part.tool_name, part.args)
        for msg in result.all_messages()
        for part in msg.parts
        if part.part_kind == "tool-call" and part.tool_name != "final_result"
    ]
    return answer, places, tool_calls


if __name__ == "__main__":
    question = sys.argv[1] if len(sys.argv) > 1 else \
        "Find cafes with wifi within 500 m of Birmingham New Street"
    answer, places, calls = ask(question, HybridSearcher())

    print("\nTool calls the agent made:")
    for name, args in calls:
        print(f"  {name}({args})")
    print(f"\n{answer.summary}\n")
    for p in places:
        print(f"  {p.distance_m:5d} m  {p.name:<35} step-free: {p.step_free}")
    for c in answer.caveats:
        print(f"  Note: {c}")