# ui_interactive/app.py
import ast
import time
import logging
import numpy as np
import streamlit as st
import plotly.graph_objects as go

from engine.models import TargetFunction
from engine.optimizers.gradient_methods import GradientDescent
from engine.optimizers.population_based import GeneticAlgorithm
from engine.initializers import RandomInitializer, HaltonInitializer
from engine.strategies.selection import (ElitismSelection, TournamentSelection, 
                                         RouletteWheelSelection, RankSelection)
from engine.strategies.crossover import UniformCrossover, NonUniformCrossover
from engine.strategies.mutation import RealCodedMutation

# --- Logging Setup ---
ui_logger = logging.getLogger("ui_app")
ui_logger.setLevel(logging.INFO)
# Prevent duplicate handlers on Streamlit reruns
if not ui_logger.handlers:
    handler = logging.FileHandler("ui.log")
    handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] UI: %(message)s'))
    ui_logger.addHandler(handler)

# --- Initialization ---
st.set_page_config(layout="wide", page_title="Optimization Engine")

if 'phase' not in st.session_state:
    ui_logger.info("Initializing new user session.")
    st.session_state.phase = 'INPUT'
if 'results' not in st.session_state:
    st.session_state.results = None
if 'f_history' not in st.session_state:
    st.session_state.f_history =[]
if 'target' not in st.session_state:
    st.session_state.target = None

# --- UI Helpers ---
def reset_optimization():
    ui_logger.info("User clicked 'New Optimization'. Resetting application state.")
    st.session_state.phase = 'INPUT'
    st.session_state.results = None
    st.session_state.f_history =[]

def parse_tuple_string(s):
    try:
        val = ast.literal_eval(s)
        if isinstance(val, (tuple, list)):
            return list(val)
        return None
    except:
        return None

# --- Sidebar: Method Specific Options ---
with st.sidebar:
    st.title("Method Specific Options")
    
    # Store method in session state directly via a selectbox in the sidebar or read from it
    # We will keep Method selection here to drive the dynamic options below
    method = st.selectbox("Optimisation Method", ["Gradient Descent", "Genetic Algorithm"], key="method")
    
    max_iter = st.number_input("Max Iterations", value=100)
    tol = st.number_input("Tolerance (\u03B5)", value=1e-4, format="%.1e")
    
    kwargs = {}
    
    if method == "Gradient Descent":
        kwargs['learning_rate'] = st.number_input("Learning Rate", value=0.1)
        kwargs['stopping_criterion'] = st.selectbox("Stopping Criterion",['gradient_norm', 'step_size'])
        kwargs['use_line_search'] = st.checkbox("Backtracking Line Search")
        kwargs['use_ravine'] = st.checkbox("Use Ravine Method")
        if kwargs['use_ravine']:
            kwargs['ravine_step_size'] = st.number_input("Ravine Step Size", value=0.5)
            
    elif method == "Genetic Algorithm":
        kwargs['population_size'] = st.number_input("Population Size", value=50)
        kwargs['stopping_criterion'] = st.selectbox("Stopping Criterion",['stagnation', 'degeneration', 'max_generations'])
        
        init_choice = st.selectbox("Initializer", ["Random", "Halton"])
        kwargs['initializer'] = RandomInitializer() if init_choice == "Random" else HaltonInitializer()
        
        sel_choice = st.selectbox("Selection",["Tournament", "Elitism", "Roulette", "Rank"])
        if sel_choice == "Tournament":
            kwargs['selection_strategy'] = TournamentSelection(tournament_size=3)
        elif sel_choice == "Elitism":
            kwargs['selection_strategy'] = ElitismSelection()
        elif sel_choice == "Roulette":
            kwargs['selection_strategy'] = RouletteWheelSelection()
        else:
            kwargs['selection_strategy'] = RankSelection()
            
        cross_choice = st.selectbox("Crossover",["Uniform", "Non-Uniform"])
        kwargs['crossover_strategy'] = UniformCrossover() if cross_choice == "Uniform" else NonUniformCrossover()
        
        kwargs['mutation_strategy'] = RealCodedMutation(sigma=0.2)
        
        st.markdown("**Coefficients**")
        col1, col2, col3 = st.columns(3)
        kwargs['phi_sel'] = col1.number_input("\u03D5 sel", value=0.2, step=0.1)
        kwargs['phi_cross'] = col2.number_input("\u03D5 cross", value=0.6, step=0.1)
        kwargs['phi_mut'] = col3.number_input("\u03D5 mut", value=0.2, step=0.1)

# --- Top Main Area (Input / Progress / Results) ---
control_area = st.container(border=True)

with control_area:
    if st.session_state.phase == 'INPUT':
        c1, c2, c3 = st.columns(3)
        func_str = c1.text_input("Target Function f(x)", value="x**2 + y**2")
        bounds_str = c2.text_input("Bounds", value="(-5, 5), (-5, 5)")
        start_str = c3.text_input("Starting Point (GD only)", value="4.0, 4.0", disabled=(method != "Gradient Descent"))
        
        if st.button("Start Optimization", type="primary"):
            ui_logger.info(f"User initiated 'Start Optimization'. Target: '{func_str}', Method: '{method}', Max Iter: {max_iter}, Tol: {tol}")
            bounds = parse_tuple_string(bounds_str)
            if bounds is None:
                ui_logger.warning(f"User input invalid bounds format: {bounds_str}")
                st.error("Invalid bounds format. Please use format: (-5, 5), (-5, 5)")
            else:
                try:
                    ui_logger.info("Initializing TargetFunction and Optimizer engine.")
                    target = TargetFunction(func_str, bounds=bounds)
                    st.session_state.target = target
                    
                    if method == "Gradient Descent":
                        start_pos = np.array(parse_tuple_string(start_str))
                        ui_logger.info(f"Configuring Gradient Descent. Start Pos: {start_pos}, Parameters: {kwargs}")
                        opt = GradientDescent(target, start_pos=start_pos, **kwargs)
                    else:
                        ui_logger.info(f"Configuring Genetic Algorithm. Parameters: {kwargs}")
                        opt = GeneticAlgorithm(target, **kwargs)
                    
                    st.session_state.optimizer = opt
                    st.session_state.max_iter = max_iter
                    st.session_state.tol = tol
                    st.session_state.phase = 'COMPUTING'
                    ui_logger.info("Transitioning to COMPUTING phase.")
                    st.rerun()
                    
                except Exception as e:
                    ui_logger.error(f"Initialization failed due to exception: {e}", exc_info=True)
                    st.error(f"Initialization failed: {e}")

    elif st.session_state.phase == 'COMPUTING':
        st.markdown("### Optimizing...")
        ui_logger.info("Rendering Progress Bar and starting optimization run loop.")
        progress_bar = st.progress(0.0)
        
        opt = st.session_state.optimizer
        max_iter = st.session_state.max_iter
        
        def update_progress(iteration):
            progress_bar.progress(min(iteration / max_iter, 1.0))
            
        try:
            # Run optimization (blocking, but updates progress bar via callback)
            t0 = time.time()
            results = opt.run(max_iter=max_iter, tol=st.session_state.tol, callback=update_progress)
            t_elapsed = time.time() - t0
            ui_logger.info(f"Optimization loop finished in {t_elapsed:.4f} seconds. Converged: {results.converged}.")
            
            # Precompute convergence history to prevent lag during animation
            if st.session_state.target is not None and results is not None and results.history is not None:
                ui_logger.info("Precomputing f(x) convergence history for UI animation.")
                f_hist =[]
                for step in results.history:
                    f_vals =[st.session_state.target.evaluate(p) for p in step["population"]]
                    f_hist.append(np.min(f_vals))
                
                st.session_state.results = results
                st.session_state.f_history = f_hist
                st.session_state.phase = 'RESULTS'
                ui_logger.info("Transitioning to RESULTS phase.")
                st.rerun()
            else:
                ui_logger.error("Optimization results are invalid or missing. Aborting results render.")
                st.error("Optimization failed to produce results. Please try again.")
                st.button("Return to Setup", on_click=reset_optimization)

        except Exception as e:
            # Catch mathematical/evaluation errors gracefully
            ui_logger.error(f"Optimization crashed during execution: {e}", exc_info=True)
            st.error("Invalid expression or bounds mismatch. Please check your math syntax.")
            st.warning("Hint: Did you forget a multiplication sign? (e.g., writing `xy` instead of `x*y` creates a 3rd variable, which breaks 2D coordinate inputs).")
            st.button("Return to Setup", on_click=reset_optimization)

    elif st.session_state.phase == 'RESULTS':
        res = st.session_state.results
        
        if res is not None and st.session_state.target is not None:
            c1, c2 = st.columns([3, 1])
            c1.markdown(f"**Optimization Complete** ({st.session_state.method})")
            c1.write(f"Converged: `{res.converged}` in `{res.iterations}` steps. Execution time: `{res.execution_time:.4f}s`")
            
            # Access results safely
            target = st.session_state.target
            best_idx = np.argmin([target.evaluate(p) for p in res.final_population])
            
            x_vals = res.final_population[best_idx]
            x_str = ", ".join([f"{val:.4f}" for val in x_vals])
            
            c1.write(f"**Final x:** `[{x_str}]` | **Final f(x):** `{res.final_f:.6f}`")
            
            c2.button("New Optimization", on_click=reset_optimization, use_container_width=True)
            ui_logger.info(f"Rendered RESULTS box successfully. Final best f(x): {res.final_f:.6f}")
        else:
            ui_logger.warning("Attempted to render RESULTS phase, but session state data was missing.")
            st.warning("Results data missing. Please run the optimization again.")

# --- Graphs Area ---
if st.session_state.phase == 'RESULTS' and st.session_state.results and st.session_state.target:
    res = st.session_state.results
    target = st.session_state.target
    
    # Animation Slider
    frame = st.slider("Generation / Iteration", 0, len(res.history) - 1, 0)
    ui_logger.info(f"Rendering visualization for frame: {frame}/{len(res.history) - 1}")
    
    col_graph1, col_graph2 = st.columns(2)
    
    # 1. Function Contour Plot
    with col_graph1:
        bounds = target.bounds
        # Fallback to default if bounds are messed up
        if not bounds or len(bounds) != 2:
            ui_logger.warning("Contour plotting skipped: Bounds are not strictly 2-dimensional.")
            st.warning("Contour plotting requires exactly 2 dimensions.")
        else:
            x_range = np.linspace(bounds[0][0], bounds[0][1], 100)
            y_range = np.linspace(bounds[1][0], bounds[1][1], 100)
            X, Y = np.meshgrid(x_range, y_range)
            
            # Vectorized evaluation requires reshaping or careful handling based on TargetFunction logic
            try:
                Z = target.evaluate([X, Y])
                if np.isscalar(Z):
                    Z = np.full_like(X, Z, dtype=float)
            except Exception as e:
                ui_logger.debug(f"Vectorized contour evaluation failed: {e}. Falling back to iterative evaluation.")
                # Fallback iterative evaluation if lambda fails on meshgrid
                Z = np.zeros_like(X)
                for i in range(X.shape[0]):
                    for j in range(X.shape[1]):
                        Z[i,j] = target.evaluate([X[i,j], Y[i,j]])
            
            fig_contour = go.Figure(data=[go.Contour(x=x_range, y=y_range, z=Z, colorscale='Viridis')])
            
            # Extract current population points
            current_pop = res.history[frame]["population"]
            px = current_pop[:, 0]
            py = current_pop[:, 1]
            
            # Add scatter for population/point
            fig_contour.add_trace(go.Scatter(x=px, y=py, mode='markers', marker=dict(color='red', size=8)))
            
            fig_contour.update_layout(title="Search Space", width=600, height=500)
            st.plotly_chart(fig_contour, use_container_width=True)
            
    # 2. Convergence Rate Plot
    with col_graph2:
        f_hist = st.session_state.f_history
        iters = list(range(len(f_hist)))
        
        fig_conv = go.Figure()
        fig_conv.add_trace(go.Scatter(x=iters, y=f_hist, mode='lines', line=dict(color='blue', width=2), name='f(x)'))
        fig_conv.add_trace(go.Scatter(x=[frame], y=[f_hist[frame]], mode='markers', marker=dict(color='red', size=10), showlegend=False))
        
        fig_conv.update_layout(title="Convergence Rate", xaxis_title="Iteration", yaxis_title="Best f(x)", width=600, height=500)
        st.plotly_chart(fig_conv, use_container_width=True)