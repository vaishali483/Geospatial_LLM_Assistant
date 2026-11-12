import pandas as pd
import streamlit as st

from search import HybridSearcher

st.set_page_config(page_title="Geospatial LLM Assistant", layout="wide")


@st.cache_resource  # load the data, indexes and embedding model only once
def get_searcher():
    return HybridSearcher()


searcher = get_searcher()
st.title("Geospatial LLM Assistant")
st.caption("OpenStreetMap data for the West Midlands. Distances are computed in code; "
           "the LLM only reads the question and explains the results.")

mode = st.radio("Mode", ["Search only (offline, no LLM)", "Ask the agent (needs internet + API key)"],
                horizontal=True)


def show_places(rows):
    if not rows:
        st.info("Nothing found.")
        return
    df = pd.DataFrame(rows)
    st.map(df, latitude="lat", longitude="lon", size=20)
    st.dataframe(df[["name", "category", "distance_m", "similarity", "step_free", "description", "id"]],
                 hide_index=True, width="stretch")
    unknown = (df["step_free"] == "unknown").sum()
    if unknown:
        st.caption(f"Step-free access is unknown for {unknown} of these places (not recorded in OSM).")


if mode.startswith("Ask"):
    try:
        from agent import ask  # imported here so offline mode works without an API key
    except Exception as e:
        st.error(f"Could not start the agent (is GOOGLE_API_KEY set in .env?): {e}")
        st.stop()

    question = st.text_input("Your question",
                             "Find cafes with wifi within 500 m of Birmingham New Street")
    if st.button("Ask") and question:
        with st.spinner("Thinking..."):
            try:
                answer, places, calls = ask(question, searcher)
            except Exception as e:  # e.g. missing API key or no internet
                st.error(f"The agent failed: {e}")
                st.stop()
        st.subheader("Answer")
        st.write(answer.summary)
        for c in answer.caveats:
            st.warning(c)
        show_places([p.model_dump() for p in places])
        with st.expander("Tool calls the agent made"):
            for name, args in calls:
                st.code(f"{name}({args})")
else:
    c1, c2, c3 = st.columns(3)
    near = c1.text_input("Near", "Birmingham New Street")
    radius = c2.slider("Radius (m)", 100, 2000, 500, step=100)
    query = c3.text_input("Looking for", "quiet cafe with wifi")
    c4, c5, c6 = st.columns(3)
    category = c4.text_input("Category (optional, e.g. cafe)", "")
    cuisine = c5.text_input("Cuisine (optional, e.g. indian)", "")
    step_free = c6.checkbox("Confirmed step-free only")
    anchor = searcher.store.geocode(near)
    if anchor is None:
        st.error(f"Could not find '{near}' in the map data.")
    else:
        st.caption(f"Searching around: {anchor['name']} ({anchor['category']})")
        rows = searcher.filter_then_rank(query, anchor["lat"], anchor["lon"], radius,
                                         category=category or None, cuisine=cuisine or None,
                                         step_free_only=step_free, k=10)
        show_places(rows)