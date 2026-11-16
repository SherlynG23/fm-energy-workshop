# FM Energy Workshop

> **Workshop Prototype** — This project is a learning exercise and not intended for production use.

## Goal

Predict a building's electricity consumption for the next hour and display the forecast on an interactive dashboard.

## Planned Tools

| Tool | Purpose |
|---|---|
| **Python** | Data processing, feature engineering, and forecasting model |
| **Streamlit** | Interactive dashboard to visualize consumption and predictions |
| **Neon Postgres** | Cloud database for storing historical and forecasted energy readings |

## Overview

The system will:
1. Ingest historical electricity consumption data into Neon Postgres
2. Train a forecasting model using Python to predict the next hour's consumption
3. Serve predictions through a Streamlit dashboard for real-time monitoring
