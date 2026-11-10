import os
from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st

from groundwater_rag import (
    build_search_index,
    load_summary_data,
    answer_groundwater_query,
    verify_groq_api,
    LEVEL_COL
)

# Project folder and CSV locations
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "Data"

FILES = [
    DATA_DIR / "gwl_manual_quarterly_cgwb_pb_1991_2020 (1).csv",
    DATA_DIR / "gwl_manual_quarterly_cgwb_pb_2021_2025.csv",
]

DATE_COL = "Data Acquisition Time"


@st.cache_data
def load_data():
    frames = []
    for file in FILES:
        if not file.exists():
            raise FileNotFoundError(f"CSV file not found: {file}")
        df = pd.read_csv(file, low_memory=False)
        df["Source File"] = file.name
        frames.append(df)

    data = pd.concat(frames, ignore_index=True)
    data.columns = data.columns.str.strip()

    required = ["District", DATE_COL, LEVEL_COL]
    missing = [col for col in required if col not in data.columns]
    if missing:
        raise ValueError(f"Required columns missing: {missing}")

    data["District_Original"] = data["District"].astype("string").str.strip().str.upper()
    data["District"] = data["District_Original"].replace({
        "MOHALI": "SAS NAGAR (MOHALI)",
        "SAS NAGAR": "SAS NAGAR (MOHALI)"
    })
    data[DATE_COL] = pd.to_datetime(data[DATE_COL], errors="coerce", dayfirst=True)
    data[LEVEL_COL] = pd.to_numeric(data[LEVEL_COL], errors="coerce")
    data = data.dropna(subset=["District", DATE_COL, LEVEL_COL])
    data = data[data["District"] != ""].copy()
    data["Year"] = data[DATE_COL].dt.year
    return data


st.set_page_config(
    page_title="AI Groundwater Assistant",
    page_icon="💧",
    layout="wide",
)

st.title("💧 AI Groundwater Information Assistant")
st.caption("CSE276 AI Foundations Project | Central Ground Water Board (CGWB) Punjab Dataset (1996–2024 Observations | 22 Canonical Districts)")

# Load Main Dataset
try:
    df = load_data()
except Exception as error:
    st.error(f"Unable to load data: {error}")
    st.stop()

if df.empty:
    st.warning("No valid groundwater records are available.")
    st.stop()

# Sidebar: Configuration & Filters
st.sidebar.title("⚙️ Settings & Filters")

# Groq API configuration in Sidebar
with st.sidebar.expander("🤖 AI Engine (Groq Setup)", expanded=True):
    env_key = os.getenv("GROQ_API_KEY", "")
    user_key = st.text_input(
        "Groq API Key (Optional)",
        value=env_key,
        type="password",
        help="Get a free key from https://console.groq.com. If left blank, deterministic rule-based answers will be used."
    )
    if user_key:
        is_valid, active_model, err_msg = verify_groq_api(user_key)
        if is_valid:
            st.success(f"🟢 Groq API Connected & Verified\n\n*(Active Model: `{active_model}`)*")
        else:
            st.error(f"🔴 Groq Verification Failed:\n{err_msg}")
            st.caption("Running in verified local analytics mode.")
    else:
        st.info("⚪ Groq API not configured. App is running in Local Analytics mode.")

st.sidebar.divider()
st.sidebar.header("Filter Dashboard Data")

min_year = int(df["Year"].min())
max_year = int(df["Year"].max())

selected_years = st.sidebar.slider(
    "Select year range",
    min_value=min_year,
    max_value=max_year,
    value=(min_year, max_year),
)

available_df = df[df["Year"].between(selected_years[0], selected_years[1])]
districts = sorted(available_df["District"].unique().tolist())

selected_districts = st.sidebar.multiselect(
    "Select districts",
    options=districts,
    default=districts,
)

filtered = available_df[available_df["District"].isin(selected_districts)].copy()

if filtered.empty:
    st.warning("No records match the selected filters.")
    st.stop()

# Summary cards
st.subheader("📊 Dataset Overview")
c1, c2, c3 = st.columns(3)
c1.metric("Valid Records", f"{len(filtered):,}")
c2.metric("Districts Selected", filtered["District"].nunique())
c3.metric("Average Measurement", f"{filtered[LEVEL_COL].mean():.2f} m")

st.info(
    "ℹ️ **Measurement Units & Hydrological Explanations:**\n"
    "- **Depth Unit (mbgl):** Groundwater level is measured in **meters below ground level**. Higher numbers denote deeper, more depleted water tables.\n"
    "- **Negative Values (e.g., Mukatsar -1.84 m, Pathankot -2.19 m):** Indicate **artesian conditions** (groundwater pressure head rising above ground level) or **surface waterlogging**.\n"
    "- **Extreme Values (>100 m, e.g., Sangrur 268 m, Jalandhar 250 m):** Verified observations from specialized deep exploratory piezometers or localized deep cones of depression.\n"
    "- **District Normalization:** **MOHALI** and **SAS NAGAR** are unified as **'SAS NAGAR (MOHALI)'** (1,162 observations, mean 11.31 mbgl) for consistent administrative analysis while preserving original station records."
)

# Year-wise trend
st.subheader("📈 Year-wise Groundwater Trend")
yearly = filtered.groupby("Year", as_index=False)[LEVEL_COL].mean()

fig_line = px.line(
    yearly,
    x="Year",
    y=LEVEL_COL,
    markers=True,
    title="Average Depth to Groundwater by Year (meters below ground level)",
    labels={"Year": "Year", LEVEL_COL: "Average Depth (meters)"},
)
fig_line.update_layout(hovermode="x unified")
st.plotly_chart(fig_line, use_container_width=True)

# District comparison
st.subheader("📍 District-wise Comparison")
district_summary = (
    filtered.groupby("District")[LEVEL_COL]
    .agg(["count", "mean", "min", "max"])
    .reset_index()
    .rename(
        columns={
            "count": "Records",
            "mean": "Average (m)",
            "min": "Minimum (m)",
            "max": "Maximum (m)",
        }
    )
    .sort_values("Average (m)", ascending=False)
)

fig_bar = px.bar(
    district_summary,
    x="District",
    y="Average (m)",
    title="Average Groundwater Level by District (Sorted Deepest to Shallowest)",
    hover_data=["Records", "Minimum (m)", "Maximum (m)"],
)
fig_bar.update_layout(xaxis_tickangle=-45)
st.plotly_chart(fig_bar, use_container_width=True)

# District ranking
st.subheader("🏆 District Depletion Ranking")
ranking = (
    filtered.groupby("District")[LEVEL_COL]
    .mean()
    .reset_index()
    .rename(columns={LEVEL_COL: "Average Measurement (m)"})
    .sort_values("Average Measurement (m)", ascending=False)
)

if not ranking.empty:
    col1, col2 = st.columns(2)
    highest = ranking.iloc[0]
    lowest = ranking.iloc[-1]

    col1.metric(
        "Deepest Water Level (Most Critical)",
        f"{highest['Average Measurement (m)']:.2f} m",
        help=f"District: {highest['District']}",
    )
    col2.metric(
        "Shallowest Water Level",
        f"{lowest['Average Measurement (m)']:.2f} m",
        help=f"District: {lowest['District']}",
    )

    st.dataframe(ranking.round(2), use_container_width=True, hide_index=True)
    st.download_button(
        "Download District Ranking CSV",
        ranking.to_csv(index=False).encode("utf-8"),
        file_name="district_ranking.csv",
        mime="text/csv",
    )

# Statistical table
st.subheader("📋 Detailed Statistical Table")
st.dataframe(district_summary.round(2), use_container_width=True, hide_index=True)

# Filtered records view
with st.expander("🔍 View Raw Filtered Records"):
    st.dataframe(filtered, use_container_width=True, hide_index=True)


# ==========================================
# RAG CHATBOT SECTION
# ==========================================
st.divider()
st.header("💬 Source-Grounded AI Groundwater Assistant")
st.write(
    "Ask any question about Punjab groundwater. "
    "The assistant uses **Hybrid RAG** (deterministic statistical computation + semantic vector search with Groq LLM)."
)

# Suggested quick questions
st.write("**Suggested Questions:**")
quick_cols = st.columns(4)
sample_q1 = "Which district has the highest groundwater measurement?"
sample_q2 = "What is the average groundwater level in Ludhiana?"
sample_q3 = "Show historical groundwater trend for Punjab"
sample_q4 = "What is the groundwater depth in village Hussainpura?"

if quick_cols[0].button("🏆 Highest Groundwater Depth"):
    st.session_state["submitted_query"] = sample_q1
if quick_cols[1].button("📍 Ludhiana District Status"):
    st.session_state["submitted_query"] = sample_q2
if quick_cols[2].button("📈 Punjab Yearly Trend"):
    st.session_state["submitted_query"] = sample_q3
if quick_cols[3].button("🏡 Village Hussainpura Status"):
    st.session_state["submitted_query"] = sample_q4


@st.cache_resource(show_spinner="Initializing Groundwater Semantic Index & Embeddings...")
def load_rag_components():
    raw_data, corpus_df, model, embeddings = build_search_index()
    d_sum, y_sum = load_summary_data()
    return raw_data, corpus_df, model, embeddings, d_sum, y_sum


try:
    rag_data, rag_corpus, rag_model, rag_embeddings, dist_sum_df, yr_sum_df = load_rag_components()

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display past conversation
    for idx, msg in enumerate(st.session_state.chat_history):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("figure") is not None:
                st.plotly_chart(msg["figure"], use_container_width=True, key=f"hist_fig_{idx}")
            if msg.get("engine"):
                st.caption(f"⚙️ **Engine:** {msg['engine']}")
            if msg.get("sources"):
                with st.expander("📚 Verified Grounding Sources"):
                    for s in msg["sources"]:
                        st.markdown(f"- `{s}`")
            if msg.get("records") is not None and not msg["records"].empty:
                with st.expander("📄 Supporting Data Records"):
                    display_cols = [
                        c for c in [
                            "District", "Station", "Tehsil", "Block", "Village",
                            "Mean_Depth", "Min_Depth", "Max_Depth", "Total_Records",
                            "Observations", "Stations", "Year", "Data Acquisition Time",
                            LEVEL_COL, "Source File", "Similarity", "mean", "min", "max",
                            "Average (mbgl)", "Average Depth (mbgl)", "Min (m)", "Max (m)", "Records"
                        ] if c in msg["records"].columns
                    ]
                    st.dataframe(msg["records"][display_cols].head(15), use_container_width=True, hide_index=True)

    # Input handling
    user_input = st.chat_input("Ask a question about groundwater, districts, or villages...")
    
    # Check if a quick button was clicked
    if "submitted_query" in st.session_state and st.session_state["submitted_query"]:
        user_input = st.session_state["submitted_query"]
        st.session_state["submitted_query"] = None

    if user_input:
        # Display user question
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Generate Assistant response
        with st.chat_message("assistant"):
            with st.spinner("Analyzing CGWB datasets and generating grounded response..."):
                active_key = user_key if user_key else None
                response = answer_groundwater_query(
                    query=user_input,
                    raw_df=rag_data,
                    corpus=rag_corpus,
                    district_summary_df=dist_sum_df,
                    yearly_summary_df=yr_sum_df,
                    model=rag_model,
                    embeddings=rag_embeddings,
                    api_key=active_key
                )

                st.markdown(response["answer"])

                if response.get("figure") is not None:
                    st.plotly_chart(response["figure"], use_container_width=True, key=f"curr_fig_{len(st.session_state.chat_history)}")

                st.caption(f"⚙️ **Engine:** {response['engine']}")

                if response.get("sources"):
                    with st.expander("📚 Verified Grounding Sources", expanded=False):
                        for s in response["sources"]:
                            st.markdown(f"- `{s}`")

                if response.get("records") is not None and not response["records"].empty:
                    with st.expander("📄 Supporting Data Records", expanded=False):
                        disp_cols = [
                            c for c in [
                                "District", "Station", "Tehsil", "Block", "Village",
                                "Mean_Depth", "Min_Depth", "Max_Depth", "Total_Records",
                                "Observations", "Stations", "Year", "Data Acquisition Time",
                                LEVEL_COL, "Source File", "Similarity", "mean", "min", "max",
                                "Average (mbgl)", "Average Depth (mbgl)", "Min (m)", "Max (m)", "Records"
                            ] if c in response["records"].columns
                        ]
                        st.dataframe(response["records"][disp_cols].head(15), use_container_width=True, hide_index=True)

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": response["answer"],
            "engine": response["engine"],
            "sources": response.get("sources", []),
            "records": response.get("records"),
            "figure": response.get("figure")
        })

    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()

except Exception as err:
    st.error("Could not load the AI Assistant components.")
    st.exception(err)
