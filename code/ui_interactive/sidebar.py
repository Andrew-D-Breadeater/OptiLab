"""
Sidebar widgets and strategy construction.

`render_sidebar()` is the single entry point. It returns
``(method, max_iter, kwargs)`` where ``kwargs`` is the dict that the
INPUT-phase Start button passes to the optimizer constructor (including
the constructed ``stopping_criterion`` instance).
"""
import sympy as sp
import streamlit as st

from engine.initializers import RandomInitializer, HaltonInitializer
from engine.strategies.selection import (ElitismSelection, TournamentSelection,
                                         RouletteWheelSelection, RankSelection)
from engine.strategies.crossover import UniformCrossover, NonUniformCrossover
from engine.strategies.mutation import RealCodedMutation
from engine.strategies.projections import (
    NoProjection, NonNegativeProjection, BoxProjection,
    HyperplaneProjection, HalfSpaceProjection, SphereProjection,
    CustomNonlinearProjection,
)
from engine.strategies.stopping import (
    StepSizeCriterion, GradientNormCriterion,
    StagnationCriterion, DegenerationCriterion, MaxGenerationsCriterion,
)
from engine.strategies.step_size import (
    FixedStepSize, BacktrackingLineSearch, ExactLineSearch,
)
from engine.strategies.gd_step import VanillaGradientStep, RavineStep, OneStepRavineStep

from ui_interactive.session import parse_tuple_string


def render_sidebar():
    with st.sidebar:
        st.title("Method Specific Options")

        method = st.selectbox("Optimisation Method",
                              ["Gradient Descent", "Newton's Method", "Genetic Algorithm"],
                              key="method")

        max_iter = st.number_input("Max Iterations", value=100)

        if method in ["Gradient Descent", "Newton's Method"]:
            kwargs = _render_traditional_sidebar(method)
        else:
            kwargs = _render_genetic_sidebar()

    return method, max_iter, kwargs


def _render_traditional_sidebar(method):
    kwargs = {}
    default_lr = 0.1 if method == "Gradient Descent" else 1.0

    use_ravine = (method == "Gradient Descent" and st.checkbox("Use Ravine Method"))

    if use_ravine:
        variant = st.selectbox(
            "Ravine Variant",
            ["Single-step (classical)", "Full descent (textbook)"],
        )

        st.markdown("**Gradient Descent step**")
        gd_strat = _render_step_size_picker(
            key_prefix="gd",
            default_lr=0.001,         # small inner GD step works for narrow ravines
            include_exact=True,
        )
        if variant == "Full descent (textbook)":
            inner_max = st.number_input("GD Max Iterations", value=30, min_value=1,
                                        key="gd_inner_max")
            stop_only_max = st.checkbox(
                "Stop only by max iterations (skip GD tolerance)",
                value=False, key="gd_stop_only_max",
            )
            if stop_only_max:
                inner_tol = None
            else:
                inner_tol = st.number_input("GD Stopping Tolerance", value=1e-4,
                                            format="%.1e", key="gd_inner_tol")

        st.markdown("**Ravine extrapolation step**")
        ravine_strat = _render_step_size_picker(
            key_prefix="ravine",
            default_lr=0.5,
            include_exact=False,       # exact line search on a heuristic ravine direction is dubious
            fixed_label="Ravine Step Size",
        )
        ravine_shift = st.number_input("Ravine Shift (v¹ bootstrap)", value=0.5,
                                       key="ravine_shift")

        kwargs['step_size_strategy'] = gd_strat   # inner VanillaGradientStep reads this
        if variant == "Single-step (classical)":
            kwargs['step_strategy'] = OneStepRavineStep(
                inner_strategy=VanillaGradientStep(),
                ravine_step_size_strategy=ravine_strat,
                ravine_shift=ravine_shift,
            )
        else:
            kwargs['step_strategy'] = RavineStep(
                inner_strategy=VanillaGradientStep(),
                ravine_step_size_strategy=ravine_strat,
                ravine_shift=ravine_shift,
                inner_tol=inner_tol,
                inner_max_iter=int(inner_max),
            )
    else:
        kwargs['step_size_strategy'] = _render_step_size_picker(
            key_prefix="main", default_lr=default_lr, include_exact=True,
        )

    crit_choice = st.selectbox("Stopping Criterion", ['gradient_norm', 'step_size'])
    crit_tol = st.number_input("Stopping Tolerance", value=1e-4, format="%.1e")
    if crit_choice == 'gradient_norm':
        kwargs['stopping_criterion'] = GradientNormCriterion(tol=crit_tol)
    else:
        kwargs['stopping_criterion'] = StepSizeCriterion(tol=crit_tol)

    if method == "Gradient Descent":
        st.markdown("---")
        st.markdown("**Constraint / Projection**")
        kwargs['projection_strategy'] = _render_projection_picker()

    return kwargs


def _render_step_size_picker(key_prefix, default_lr, include_exact=True,
                             fixed_label="Learning Rate"):
    """
    Render a Fixed | Backtracking [| Exact] selector with mode-specific
    parameter inputs and return the constructed StepSizeStrategy.

    All widget keys are prefixed by ``key_prefix`` so multiple pickers can
    coexist on the same page (e.g. inner GD step + ravine extrapolation).
    """
    options = ["Fixed", "Backtracking"]
    if include_exact:
        options.append("Exact")
    label = "Step Size" if key_prefix == "main" else f"{key_prefix.capitalize()} Step Size Mode"
    choice = st.selectbox(label, options, key=f"{key_prefix}_mode")
    if choice == "Fixed":
        lr = st.number_input(fixed_label, value=default_lr, format="%.4f",
                             key=f"{key_prefix}_lr")
        decay = st.number_input("Decay Rate", value=1.0, min_value=0.0, max_value=1.0,
                                key=f"{key_prefix}_decay")
        return FixedStepSize(learning_rate=lr, decay_rate=decay)
    if choice == "Backtracking":
        alpha0 = st.number_input("Initial α (backtracking)", value=1.0,
                                 key=f"{key_prefix}_alpha0")
        return BacktrackingLineSearch(alpha0=alpha0)
    return ExactLineSearch()


def _render_projection_picker():
    proj_type = st.selectbox(
        "Boundary Projection",
        ["None", "Non-Negative", "Box (Bounds)", "Hyperplane", "Half-Space", "Sphere", "Custom Non-linear"],
    )

    if proj_type == "None":
        return NoProjection()

    if proj_type == "Non-Negative":
        return NonNegativeProjection()

    if proj_type == "Box (Bounds)":
        bounds_str = st.session_state.get('form_bounds', "(-5, 5), (-5, 5)")
        parsed_bounds = parse_tuple_string(bounds_str)
        if parsed_bounds:
            return BoxProjection(parsed_bounds)
        st.sidebar.error("Invalid Bounds format in main area.")
        return NoProjection()

    if proj_type in ["Hyperplane", "Half-Space"]:
        c_str = st.text_input("Normal Vector (c)", value="1.0, 1.0")
        b_val = st.number_input("Scalar (b)", value=4.0)
        c_vec = parse_tuple_string(c_str)
        if not c_vec:
            st.sidebar.error("Invalid normal vector format.")
            return NoProjection()
        if proj_type == "Hyperplane":
            return HyperplaneProjection(c=c_vec, b=b_val)
        return HalfSpaceProjection(c=c_vec, b=b_val)

    if proj_type == "Sphere":
        center_str = st.text_input("Center", value="0.0, 0.0")
        r_val = st.number_input("Radius", value=2.0, min_value=0.01)
        center_vec = parse_tuple_string(center_str)
        if not center_vec:
            st.sidebar.error("Invalid center format.")
            return NoProjection()
        return SphereProjection(center=center_vec, radius=r_val)

    if proj_type == "Custom Non-linear":
        st.info("Write one constraint per line (e.g., `y - x**3 >= 0`).")
        c_text = st.text_area("Constraints", value="y - x**2 >= 0\nx >= 0\ny >= 0")
        c_lines = [line.strip() for line in c_text.split('\n') if line.strip()]

        try:
            expr_str = st.session_state.get('form_expr', 'x')
            sympy_expr = sp.sympify(expr_str)
            found_symbols = sympy_expr.free_symbols

            raw_bounds_str = st.session_state.get('form_bounds', '(-5, 5), (-5, 5)')
            parsed_bounds = parse_tuple_string(raw_bounds_str)

            if parsed_bounds and len(parsed_bounds) == 2 and len(found_symbols) == 1:
                all_symbols = {sp.Symbol('x'), sp.Symbol('y')}
            else:
                all_symbols = found_symbols

            variables = sorted([s.name for s in all_symbols])
            return CustomNonlinearProjection(c_lines, variables)
        except Exception as e:
            st.sidebar.error(f"Waiting for valid target function... ({e})")
            return NoProjection()

    return NoProjection()


def _render_genetic_sidebar():
    kwargs = {}
    kwargs['population_size'] = st.number_input("Population Size", value=50)

    crit_choice = st.selectbox("Stopping Criterion",
                               ['stagnation', 'degeneration', 'max_generations'])
    if crit_choice == 'stagnation':
        patience = st.number_input("Patience (generations)", value=15, min_value=1)
        kwargs['stopping_criterion'] = StagnationCriterion(patience=int(patience))
    elif crit_choice == 'degeneration':
        deg_tol = st.number_input("Degeneration Tolerance", value=0.05, format="%.2e")
        kwargs['stopping_criterion'] = DegenerationCriterion(tol=deg_tol)
    else:
        kwargs['stopping_criterion'] = MaxGenerationsCriterion()

    init_choice = st.selectbox("Initializer", ["Random", "Halton"])
    kwargs['initializer'] = RandomInitializer() if init_choice == "Random" else HaltonInitializer()

    sel_choice = st.selectbox("Selection", ["Tournament", "Elitism", "Roulette", "Rank"])
    if sel_choice == "Tournament":
        kwargs['selection_strategy'] = TournamentSelection(tournament_size=3)
    elif sel_choice == "Elitism":
        kwargs['selection_strategy'] = ElitismSelection()
    elif sel_choice == "Roulette":
        kwargs['selection_strategy'] = RouletteWheelSelection()
    else:
        kwargs['selection_strategy'] = RankSelection()

    cross_choice = st.selectbox("Crossover", ["Uniform", "Non-Uniform"])
    kwargs['crossover_strategy'] = UniformCrossover() if cross_choice == "Uniform" else NonUniformCrossover()

    kwargs['mutation_strategy'] = RealCodedMutation(sigma=0.2)

    st.markdown("**Coefficients**")
    col1, col2, col3 = st.columns(3)
    kwargs['phi_sel'] = col1.number_input("ϕ sel", value=0.2, step=0.1)
    kwargs['phi_cross'] = col2.number_input("ϕ cross", value=0.6, step=0.1)
    kwargs['phi_mut'] = col3.number_input("ϕ mut", value=0.2, step=0.1)

    return kwargs
