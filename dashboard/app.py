"""
FM Energy Dashboard
-------------------
Reads meter_readings_source from Neon PostgreSQL (via the HTTP SQL API)
and displays historical hourly electricity consumption.

Run from the project root:
    ..\\vibe\\Scripts\\streamlit.exe run dashboard/app.py --server.headless true
"""

import os
import re
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from dotenv import load_dotenv

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FM Energy Dashboard",
    page_icon="⚡",
    layout="wide",
)

# ── Load environment ──────────────────────────────────────────────────────────
load_dotenv(Path(__file__).parent.parent / ".env")

raw_url = os.environ["DATABASE_URL"]
conn_str = re.sub(r"[&?]channel_binding=require", "", raw_url)
host = re.search(r"@([^/]+)/", conn_str).group(1)
ENDPOINT = f"https://{host}/sql"
HEADERS = {
    "Content-Type": "application/json",
    "Neon-Connection-String": conn_str,
}

# ── Colour palette — one distinct colour per meter ───────────────────────────
METER_COLORS = {
    "METER_01": "#00C0F3",   # cyan-blue
    "METER_02": "#FF6B35",   # warm orange
    "METER_03": "#6BCB77",   # green
}
TOTAL_COLOR = "#A78BFA"      # soft violet for the combined total

# ── Data fetch ────────────────────────────────────────────────────────────────

@st.cache_data(ttl=60, show_spinner="Fetching readings from Neon …")
def fetch_readings() -> pd.DataFrame:
    """Return all rows from meter_readings_source as a DataFrame."""
    result = requests.post(
        ENDPOINT,
        headers=HEADERS,
        json={
            "query": """
                SELECT
                    timestamp,
                    meter_id,
                    building_id,
                    electricity_kwh
                FROM meter_readings_source
                ORDER BY timestamp, meter_id
            """
        },
        timeout=30,
    )
    result.raise_for_status()
    data = result.json()

    rows = data.get("rows", [])
    if not rows:
        return pd.DataFrame(
            columns=["timestamp", "meter_id", "building_id", "electricity_kwh"]
        )

    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["electricity_kwh"] = pd.to_numeric(df["electricity_kwh"])
    return df


# ── Chart helpers ─────────────────────────────────────────────────────────────

def meter_chart(meter_df: pd.DataFrame, meter_id: str) -> go.Figure:
    """Line chart for a single meter."""
    color = METER_COLORS.get(meter_id, "#FFFFFF")
    fig = px.line(
        meter_df,
        x="timestamp",
        y="electricity_kwh",
        labels={"timestamp": "Time", "electricity_kwh": "kWh"},
    )
    fig.update_traces(line_color=color, line_width=1.5)
    fig.update_layout(
        margin=dict(t=10, b=10, l=10, r=10),
        plot_bgcolor="#0F1117",
        paper_bgcolor="#0F1117",
        font_color="#FAFAFA",
        xaxis=dict(showgrid=False, tickfont_size=10),
        yaxis=dict(showgrid=True, gridcolor="#2A2A3A", tickfont_size=10),
        height=240,
        showlegend=False,
    )
    return fig


def total_chart(total_df: pd.DataFrame) -> go.Figure:
    """Line chart for the summed total across all meters."""
    fig = px.line(
        total_df,
        x="timestamp",
        y="electricity_kwh",
        labels={"timestamp": "Time", "electricity_kwh": "Total kWh"},
    )
    fig.update_traces(line_color=TOTAL_COLOR, line_width=2)
    fig.update_layout(
        margin=dict(t=10, b=10, l=10, r=10),
        plot_bgcolor="#0F1117",
        paper_bgcolor="#0F1117",
        font_color="#FAFAFA",
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor="#2A2A3A"),
        height=280,
        showlegend=False,
    )
    return fig


# ── App layout ────────────────────────────────────────────────────────────────

# Title
st.markdown(
    "<h1 style='text-align:center; color:#00C0F3;'>⚡ FM Energy Dashboard</h1>",
    unsafe_allow_html=True,
)
st.caption(
    "Historical hourly electricity consumption — data sourced from Neon PostgreSQL"
)
st.divider()

# Fetch data
df = fetch_readings()

if df.empty:
    st.error("No data returned from the database. Check your DATABASE_URL in .env.")
    st.stop()

meters = sorted(df["meter_id"].unique())

# ── Section 1: per-meter charts ───────────────────────────────────────────────
st.subheader("Hourly Consumption by Meter  *(Historical)*")

cols = st.columns(len(meters))
for col, meter_id in zip(cols, meters):
    meter_df = df[df["meter_id"] == meter_id].copy()
    building = meter_df["building_id"].iloc[0]
    with col:
        st.markdown(
            f"<p style='font-size:13px; color:{METER_COLORS.get(meter_id,'#FFF')};'>"
            f"<b>{meter_id}</b> · {building}</p>",
            unsafe_allow_html=True,
        )
        st.plotly_chart(
            meter_chart(meter_df, meter_id),
            use_container_width=True,
            key=f"chart_{meter_id}",
        )

st.divider()

# ── Section 2: total chart ────────────────────────────────────────────────────
st.subheader("Total Electricity Usage — All Meters Combined  *(Historical)*")

total_df = (
    df.groupby("timestamp", as_index=False)["electricity_kwh"]
    .sum()
    .sort_values("timestamp")
)

# Quick KPI strip above the total chart
kpi1, kpi2, kpi3 = st.columns(3)
kpi1.metric("Total kWh (all time)", f"{df['electricity_kwh'].sum():,.0f} kWh")
kpi2.metric("Peak hour (single meter)", f"{df['electricity_kwh'].max():,.2f} kWh")
kpi3.metric("Average hourly (per meter)", f"{df['electricity_kwh'].mean():,.2f} kWh")

st.plotly_chart(total_chart(total_df), use_container_width=True, key="chart_total")

st.divider()

# ── Section 3: Prediction placeholder ────────────────────────────────────────
st.subheader("🔮 Next-Hour Prediction")
st.info(
    "**Coming soon.** A trained Gradient Boosting model will predict each meter's "
    "consumption for the next hour using lag features (1 h, 2 h, 3 h, 24 h, 168 h).",
    icon="🔮",
)

st.divider()

# ── Section 4: LLM Analysis placeholder ──────────────────────────────────────
st.subheader("🤖 LLM Energy Analysis")
st.info(
    "**Coming soon.** An OpenAI-powered analysis will summarise trends, flag anomalies, "
    "and suggest efficiency improvements based on the last 24 hours of readings.",
    icon="🤖",
)

st.divider()

# ── Footer ────────────────────────────────────────────────────────────────────
st.caption(
    "Run from the project root: "
    "`..\\\\vibe\\\\Scripts\\\\streamlit.exe run dashboard/app.py --server.headless true`"
)
