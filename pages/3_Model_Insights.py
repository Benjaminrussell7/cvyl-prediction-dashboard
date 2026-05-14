from __future__ import annotations

import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1] / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import streamlit as st

import streamlit_app as dashboard


def main() -> None:
    data = dashboard.load_dashboard_data()

    st.title("Model Insights")
    dashboard.render_data_freshness()

    st.caption(
        "This page keeps validation and methodology details separate from the coaching workflow. "
        "Power Rating probabilities are the primary displayed prediction metric; ELO remains supporting context."
    )
    dashboard.render_model_insights(
        data["model_comparison_summary"],
        data["calibration"],
        data["backtest"],
    )


if __name__ == "__main__":
    main()
