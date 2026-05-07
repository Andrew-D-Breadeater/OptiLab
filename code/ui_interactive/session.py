"""
Session state, logging, and small helpers for the Streamlit UI.

Vault pattern
-------------
Streamlit garbage-collects widget state when the widget is not currently
rendered (e.g. while the COMPUTING phase replaces the INPUT form). To
survive that, every persistent input is stored under TWO keys:

* ``form_*``       — the actual widget keys; live only while the widget is on screen.
* ``persistent_*`` — vault keys that are never bound to a widget, so Streamlit
                     never collects them. Set on every "Start Optimization"
                     click and re-used as ``value=`` when the form is rendered.

Sidebar code that needs to read user input across rerenders should read the
``persistent_*`` keys, not the widget keys — widget keys may not exist yet on
the current rerender pass.
"""
import ast
import logging
import numpy as np

import streamlit as st

from engine.function_library import FunctionLibrary


# --- Logging --------------------------------------------------------------

ui_logger = logging.getLogger("ui_app")


def configure_logger() -> None:
    """Attach the file handler exactly once across Streamlit reruns."""
    ui_logger.setLevel(logging.INFO)
    if not ui_logger.handlers:
        handler = logging.FileHandler("ui.log")
        handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] UI: %(message)s'))
        ui_logger.addHandler(handler)


# --- Session state --------------------------------------------------------

def init_session_state() -> None:
    """Populate every key the rest of the UI assumes exists."""
    if 'func_lib' not in st.session_state:
        st.session_state.func_lib = FunctionLibrary()

    # Vault keys (see module docstring)
    if 'persistent_expr' not in st.session_state:
        st.session_state.persistent_expr = "x**2 + y**2"
    if 'persistent_bounds' not in st.session_state:
        st.session_state.persistent_bounds = "(-5, 5), (-5, 5)"
    if 'persistent_start' not in st.session_state:
        st.session_state.persistent_start = "4.0, 4.0"

    if 'phase' not in st.session_state:
        ui_logger.info("Initializing new user session.")
        st.session_state.phase = 'INPUT'
    if 'results' not in st.session_state:
        st.session_state.results = None
    if 'f_history' not in st.session_state:
        st.session_state.f_history = []
    if 'target' not in st.session_state:
        st.session_state.target = None
    if 'is_convex' not in st.session_state:
        st.session_state.is_convex = None
    if 'bad_point' not in st.session_state:
        st.session_state.bad_point = None


def reset_optimization() -> None:
    ui_logger.info("User clicked 'New Optimization'. Resetting application state.")
    st.session_state.phase = 'INPUT'
    st.session_state.preset_selector = "-- Custom / Manual --"
    st.session_state.results = None
    st.session_state.f_history = []
    st.session_state.is_convex = None
    st.session_state.bad_point = None


# --- Helpers --------------------------------------------------------------

def parse_tuple_string(s):
    try:
        val = ast.literal_eval(s)
        if isinstance(val, (tuple, list)):
            return list(val)
        return None
    except Exception:
        return None


def precompute_f_history(target, history) -> list:
    """Pre-evaluate best f(x) per iteration so the animation slider is snappy."""
    f_hist = []
    for step in history:
        f_vals = [target.evaluate(p) for p in step["population"]]
        f_hist.append(np.min(f_vals))
    return f_hist
