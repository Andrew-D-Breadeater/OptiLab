"""Streamlit entry point — page config, logging, session init, and phase router."""
import streamlit as st

from ui_interactive.session import configure_logger, init_session_state
from ui_interactive.sidebar import render_sidebar
from ui_interactive.phases import render_input_phase, render_computing_phase, render_results_phase


st.set_page_config(layout="wide", page_title="Optimization Engine")

configure_logger()
init_session_state()

method, max_iter, kwargs = render_sidebar()

control_area = st.container(border=True)
with control_area:
    if st.session_state.phase == 'INPUT':
        render_input_phase(method, max_iter, kwargs)
    elif st.session_state.phase == 'COMPUTING':
        render_computing_phase()
    elif st.session_state.phase == 'RESULTS':
        render_results_phase()

# RESULTS-phase visualization area is rendered inside render_results_phase().
