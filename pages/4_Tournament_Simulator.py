from __future__ import annotations

import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1] / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import streamlit as st

import streamlit_app as dashboard


def main() -> None:
    st.title("Tournament Simulator")
    dashboard.render_data_freshness()

    st.divider()
    with st.container(border=True):
        st.subheader("Coming Soon")
        st.write(
            "This page is reserved for future tournament tools. Planned functionality includes "
            "bracket simulations, championship probabilities, upset likelihood, and Monte Carlo simulations."
        )
        st.caption("No simulation logic is implemented yet.")


if __name__ == "__main__":
    main()
