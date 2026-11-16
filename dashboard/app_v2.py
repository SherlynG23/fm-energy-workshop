"""
FM Energy Dashboard v2 — Simulated Live Feed
---------------------------------------------
Starts with January 2016 data for all three meters.
Every real minute it fetches the next recorded hour from Neon (one reading per
meter) and appends it to the charts, simulating a live sensor feed.
When the last record is reached the simulation stops automatically.
A "Restart simulation" button resets everything back to 1 January 2016.

Run from the project root:
    ..\\vibe\\Scripts\\streamlit.exe run dashboard/app_v2.py --server.headless true
"""

import os
import re
import time
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from dotenv import load_dotenv

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FM Energy Dashboard v2",
    page_icon="⚡",
    layout="wide",
)

# ── Environment ───────────────────────────────────────────────────────────────
load_dotenv(Path(__file__).parent.parent / ".env")

raw_url = os.environ["DATABASE_URL"]
conn_str = re.sub(r"[&?]channel_binding=require", "", raw_url)
host = re.search(r"@([^/]+)/", conn_str).group(1)
ENDPOINT = f"https://{host}/sql"
NEON_HEADERS = {
    "Content-Type": "application/json",
    "Neon-Connection-String": conn_str,
}

# ── Constants ─────────────────────────────────────────────────────────────────
METER_COLORS = {
    "METER_01": "#00C0F3",
    "METER_02": "#FF6B35",
    "METER_03": "#6BCB77",
}
TOTAL_COLOR   = "#A78BFA"
TICK_SECONDS  = 60          # real seconds between each simulated hour fetch
JAN_END       = "2016-02-01T00:00:00+00:00"   # exclusive upper bound for initial load

# ── Neon query helper ─────────────────────────────────────────────────────────

def neon_query(query: str, params: list | None = None) -> list[dict]:
    body: dict = {"query": query}
    if params:
        body["params"] = params
    r = requests.post(ENDPOINT, headers=NEON_HEADERS, json=body, timeout=30)
    r.raise_for_status()
    return r.json().get("rows", [])


# ── Data helpers ──────────────────────────────────────────────────────────────

def load_january() -> pd.DataFrame:
    """Fetch all January 2016 readings (timestamp < 1 Feb 2016)."""
    rows = neon_query(
        """
        SELECT timestamp, meter_id, building_id, electricity_kwh
        FROM   meter_readings_source
        WHERE  timestamp < $1
        ORDER  BY timestamp, meter_id
        """,
        [JAN_END],
    )
    return _to_df(rows)


def fetch_next_hour(after_ts: str) -> pd.DataFrame:
    """Fetch the single next distinct hour that comes after after_ts (all meters)."""
    rows = neon_query(
        """
        WITH next_ts AS (
            SELECT MIN(timestamp) AS ts
            FROM   meter_readings_source
            WHERE  timestamp > $1
        )
        SELECT s.timestamp, s.meter_id, s.building_id, s.electricity_kwh
        FROM   meter_readings_source s
        JOIN   next_ts n ON s.timestamp = n.ts
        ORDER  BY s.meter_id
        """,
        [after_ts],
    )
    return _to_df(rows)


def _to_df(rows: list[dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(
            columns=["timestamp", "meter_id", "building_id", "electricity_kwh"]
        )
    df = pd.DataFrame(rows)
    df["timestamp"]       = pd.to_datetime(df["timestamp"], utc=True)
    df["electricity_kwh"] = pd.to_numeric(df["electricity_kwh"])
    return df


# ── Chart helpers ─────────────────────────────────────────────────────────────

def _base_layout(height: int) -> dict:
    return dict(
        margin=dict(t=10, b=10, l=10, r=10),
        plot_bgcolor="#0F1117",
        paper_bgcolor="#0F1117",
        font_color="#FAFAFA",
        xaxis=dict(showgrid=False, tickfont_size=10),
        yaxis=dict(showgrid=True, gridcolor="#2A2A3A", tickfont_size=10),
        height=height,
        showlegend=False,
    )


def meter_chart(meter_df: pd.DataFrame, meter_id: str) -> go.Figure:
    color = METER_COLORS.get(meter_id, "#FFFFFF")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=meter_df["timestamp"],
        y=meter_df["electricity_kwh"],
        mode="lines",
        line=dict(color=color, width=1.5),
    ))
    fig.update_layout(**_base_layout(240))
    return fig


def total_chart(df: pd.DataFrame) -> go.Figure:
    total = (
        df.groupby("timestamp", as_index=False)["electricity_kwh"]
        .sum()
        .sort_values("timestamp")
    )
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=total["timestamp"],
        y=total["electricity_kwh"],
        mode="lines",
        line=dict(color=TOTAL_COLOR, width=2),
    ))
    fig.update_layout(**_base_layout(280))
    return fig


# ── Session-state initialisation ──────────────────────────────────────────────

def _init_state():
    """Load January data and set the simulation clock. Called at startup and on restart."""
    with st.spinner("Loading January 2016 data from Neon …"):
        df = load_january()
    st.session_state.df         = df
    st.session_state.next_tick  = time.time() + TICK_SECONDS   # wall-clock target
    st.session_state.finished   = False
    st.session_state.initialized = True


if not st.session_state.get("initialized"):
    _init_state()

# ── Restart button (evaluated before charts so state is clean this run) ───────
if st.session_state.get("restart_requested"):
    st.session_state.restart_requested = False
    _init_state()

# ── Simulation tick ───────────────────────────────────────────────────────────
# If we're not finished and the wall clock has passed next_tick, fetch one more hour.

if not st.session_state.finished:
    now = time.time()
    if now >= st.session_state.next_tick:
        last_ts = st.session_state.df["timestamp"].max()
        new_rows = fetch_next_hour(last_ts.isoformat())

        if new_rows.empty:
            st.session_state.finished = True
        else:
            st.session_state.df = pd.concat(
                [st.session_state.df, new_rows], ignore_index=True
            )
            st.session_state.next_tick = time.time() + TICK_SECONDS

# ── Convenience references ────────────────────────────────────────────────────
df     = st.session_state.df
meters = sorted(df["meter_id"].unique())

# ── Simulation status ─────────────────────────────────────────────────────────
current_max_ts = df["timestamp"].max()
if st.session_state.finished:
    sim_status = "🏁 Simulation complete — all records displayed."
else:
    secs_left = max(0, int(st.session_state.next_tick - time.time()))
    sim_status = (
        f"🟢 Live simulation · showing up to **{current_max_ts.strftime('%d %b %Y %H:%M UTC')}** · "
        f"next update in **{secs_left}s**"
    )

# ── Title bar ─────────────────────────────────────────────────────────────────
st.markdown(
    "<h1 style='text-align:center; color:#00C0F3;'>⚡ FM Energy Dashboard v2</h1>",
    unsafe_allow_html=True,
)

status_col, btn_col = st.columns([5, 1])
with status_col:
    st.caption(sim_status)
with btn_col:
    if st.button("🔄 Restart simulation", use_container_width=True):
        st.session_state.restart_requested = True
        st.rerun()

st.divider()

# ── Section 1: per-meter charts ───────────────────────────────────────────────
st.subheader("Hourly Consumption by Meter")

cols = st.columns(len(meters))
for col, meter_id in zip(cols, meters):
    meter_df = df[df["meter_id"] == meter_id].copy()
    building = meter_df["building_id"].iloc[0]
    color = METER_COLORS.get(meter_id, "#FFFFFF")
    with col:
        st.markdown(
            f"<p style='font-size:13px; color:{color};'>"
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
st.subheader("Total Electricity Usage — All Meters Combined")

kpi1, kpi2, kpi3 = st.columns(3)
kpi1.metric("Total kWh displayed",        f"{df['electricity_kwh'].sum():,.0f} kWh")
kpi2.metric("Peak hour (single meter)",   f"{df['electricity_kwh'].max():,.2f} kWh")
kpi3.metric("Average hourly (per meter)", f"{df['electricity_kwh'].mean():,.2f} kWh")

st.plotly_chart(total_chart(df), use_container_width=True, key="chart_total")

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
    "`..\\\\vibe\\\\Scripts\\\\streamlit.exe run dashboard/app_v2.py --server.headless true`"
)

# ── Auto-rerun ────────────────────────────────────────────────────────────────
# Rerun once per second while the simulation is active so the countdown
# updates and the tick fires as soon as the minute is up.
if not st.session_state.finished:
    time.sleep(1)
    st.rerun()
