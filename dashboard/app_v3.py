"""
FM Energy Dashboard v3 — Simulated Live Feed + Next-Hour Prediction
--------------------------------------------------------------------
Based on v2. Adds a "Predict next hour" button in the Prediction section.
Uses the saved GradientBoostingRegressor (models/gbr_model.joblib) and the
recorded feature order (models/feature_order.json). No retraining.

For each meter the prediction is built from that meter's own revealed readings.
The minimum history required is 168 rows (one week of hourly data) to satisfy
the lag_168h feature. If fewer rows are available a clear message is shown.

Run from the project root:
    ..\\vibe\\Scripts\\streamlit.exe run dashboard/app_v3.py --server.headless true
"""

import json
import os
import re
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from dotenv import load_dotenv

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FM Energy Dashboard v3",
    page_icon="⚡",
    layout="wide",
)

# ── Environment ───────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent
load_dotenv(BASE_DIR / ".env")

raw_url = os.environ["DATABASE_URL"]
conn_str = re.sub(r"[&?]channel_binding=require", "", raw_url)
host = re.search(r"@([^/]+)/", conn_str).group(1)
ENDPOINT = f"https://{host}/sql"
NEON_HEADERS = {
    "Content-Type": "application/json",
    "Neon-Connection-String": conn_str,
}

# ── Load model and feature order (once, cached) ───────────────────────────────
@st.cache_resource
def load_model():
    model        = joblib.load(BASE_DIR / "models" / "gbr_model.joblib")
    feature_cols = json.loads((BASE_DIR / "models" / "feature_order.json").read_text())
    return model, feature_cols

MODEL, FEATURE_COLS = load_model()

# The longest lag the model needs
MAX_LAG = 168   # lag_168h → need at least 168 previous rows

# ── Constants ─────────────────────────────────────────────────────────────────
METER_COLORS = {
    "METER_01": "#00C0F3",
    "METER_02": "#FF6B35",
    "METER_03": "#6BCB77",
}
TOTAL_COLOR  = "#A78BFA"
TICK_SECONDS = 60
JAN_END      = "2016-02-01T00:00:00+00:00"

# ── Neon helper ───────────────────────────────────────────────────────────────

def neon_query(query: str, params: list | None = None) -> list[dict]:
    body: dict = {"query": query}
    if params:
        body["params"] = params
    r = requests.post(ENDPOINT, headers=NEON_HEADERS, json=body, timeout=30)
    r.raise_for_status()
    return r.json().get("rows", [])


# ── Data helpers ──────────────────────────────────────────────────────────────

def load_january() -> pd.DataFrame:
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


# ── Prediction helper ─────────────────────────────────────────────────────────

def predict_for_meter(meter_df: pd.DataFrame) -> dict:
    """
    Build lag features from meter_df and predict the next hour.

    Returns a dict with keys:
        ok        bool   – whether a prediction was produced
        kwh       float  – predicted kWh (only when ok=True)
        target_ts pd.Timestamp – the timestamp being predicted (ok=True)
        message   str    – human-readable status / error
    """
    # Sort and work on a clean series
    s = (
        meter_df.sort_values("timestamp")
        .reset_index(drop=True)["electricity_kwh"]
    )
    n = len(s)

    if n < MAX_LAG:
        needed = MAX_LAG - n
        return {
            "ok": False,
            "message": (
                f"Need {MAX_LAG} readings for the lag-168h feature; "
                f"only {n} available. "
                f"Wait for {needed} more hour(s) to be revealed."
            ),
        }

    last_ts     = meter_df.sort_values("timestamp")["timestamp"].iloc[-1]
    target_ts   = last_ts + pd.Timedelta(hours=1)

    features = {
        "hour":        target_ts.hour,
        "day_of_week": target_ts.weekday(),
        "lag_1h":      float(s.iloc[-1]),
        "lag_2h":      float(s.iloc[-2]),
        "lag_3h":      float(s.iloc[-3]),
        "lag_24h":     float(s.iloc[-24]),
        "lag_168h":    float(s.iloc[-168]),
    }

    X = pd.DataFrame([features])[FEATURE_COLS]
    kwh = float(MODEL.predict(X)[0])

    return {
        "ok":        True,
        "kwh":       kwh,
        "target_ts": target_ts,
        "message":   f"Predicted for {target_ts.strftime('%d %b %Y %H:%M UTC')}",
    }


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
    with st.spinner("Loading January 2016 data from Neon …"):
        df = load_january()
    st.session_state.df             = df
    st.session_state.next_tick      = time.time() + TICK_SECONDS
    st.session_state.finished       = False
    st.session_state.initialized    = True
    st.session_state.predictions    = {}   # meter_id → result dict


if not st.session_state.get("initialized"):
    _init_state()

if st.session_state.get("restart_requested"):
    st.session_state.restart_requested = False
    _init_state()

# ── Simulation tick ───────────────────────────────────────────────────────────
if not st.session_state.finished:
    now = time.time()
    if now >= st.session_state.next_tick:
        last_ts  = st.session_state.df["timestamp"].max()
        new_rows = fetch_next_hour(last_ts.isoformat())
        if new_rows.empty:
            st.session_state.finished = True
        else:
            st.session_state.df = pd.concat(
                [st.session_state.df, new_rows], ignore_index=True
            )
            st.session_state.next_tick = time.time() + TICK_SECONDS
            # Clear stale predictions when new data arrives
            st.session_state.predictions = {}

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
        f"🟢 Live simulation · showing up to "
        f"**{current_max_ts.strftime('%d %b %Y %H:%M UTC')}** · "
        f"next update in **{secs_left}s**"
    )

# ── Title bar ─────────────────────────────────────────────────────────────────
st.markdown(
    "<h1 style='text-align:center; color:#00C0F3;'>⚡ FM Energy Dashboard v3</h1>",
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
    color    = METER_COLORS.get(meter_id, "#FFFFFF")
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

# ── Section 3: Prediction ─────────────────────────────────────────────────────
st.subheader("🔮 Next-Hour Prediction")

st.caption(
    "Uses the saved Gradient Boosting model with lag features "
    "(1 h, 2 h, 3 h, 24 h, 168 h). Requires at least 168 revealed readings per meter."
)

if st.button("🔮 Predict next hour", use_container_width=False):
    results = {}
    for meter_id in meters:
        meter_df = df[df["meter_id"] == meter_id].copy()
        results[meter_id] = predict_for_meter(meter_df)
    st.session_state.predictions = results

predictions = st.session_state.get("predictions", {})

if predictions:
    pred_cols = st.columns(len(meters))
    for col, meter_id in zip(pred_cols, meters):
        result = predictions.get(meter_id, {})
        color  = METER_COLORS.get(meter_id, "#FFFFFF")
        with col:
            st.markdown(
                f"<p style='font-size:13px; color:{color};'><b>{meter_id}</b></p>",
                unsafe_allow_html=True,
            )
            if result.get("ok"):
                st.metric(
                    label=result["message"],
                    value=f"{result['kwh']:,.2f} kWh",
                )
            else:
                st.warning(result.get("message", "No prediction available."))
else:
    st.info(
        "Press **Predict next hour** to run the model against currently visible readings.",
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
    "`..\\\\vibe\\\\Scripts\\\\streamlit.exe run dashboard/app_v3.py --server.headless true`"
)

# ── Auto-rerun ────────────────────────────────────────────────────────────────
if not st.session_state.finished:
    time.sleep(1)
    st.rerun()
