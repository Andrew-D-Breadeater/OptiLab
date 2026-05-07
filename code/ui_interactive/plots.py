"""Pure plotting helpers — no Streamlit, no session state."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go


def _classify_ravine_frames(phases):
    """
    For Ravine* step strategies, tag each history frame as one of:
      - 'v' : extrapolation point (vⁱ)
      - 'x' : end-of-descent point (xⁱ — last 'descend' in its block)
      - 'd' : intermediate descent iterate inside a multi-step block (dⁱⱼ)

    The first untagged frame is v⁰. Later untagged frames are duplicates
    of whatever the previous tagged frame was (could be x or v depending
    on which alternation the step strategy is in) — we carry that over so
    coincident markers stay coherent.
    """
    n = len(phases)
    kinds = []
    seen_v0 = False
    last_kind = None
    for idx, phase in enumerate(phases):
        if phase is None:
            if not seen_v0:
                kinds.append('v')
                seen_v0 = True
                last_kind = 'v'
            else:
                kinds.append(last_kind or 'x')
        elif phase == 'extrapolate':
            kinds.append('v')
            last_kind = 'v'
        elif phase == 'descend':
            is_last_in_block = (idx == n - 1) or (phases[idx + 1] != 'descend')
            kind = 'x' if is_last_in_block else 'd'
            kinds.append(kind)
            last_kind = kind
        else:
            kinds.append('x')
            last_kind = 'x'
    return kinds


def build_contour_figure(target, projection_strategy, history, frame, mode):
    """
    Build the contour landscape with trajectory or population swarm overlaid.

    Args:
        target: TargetFunction (must have .bounds and .evaluate).
        projection_strategy: ProjectionStrategy or None — drives forbidden-zone shading.
        history: list of OptimisationResults.history entries.
        frame: index into history for the current animation frame.
        mode: 'trajectory' for single-agent line, 'swarm' for population scatter.
    """
    bounds = target.bounds
    if not bounds or len(bounds) != 2:
        raise ValueError("Contour plotting requires exactly 2 dimensions.")

    x_range = np.linspace(bounds[0][0], bounds[0][1], 300)
    y_range = np.linspace(bounds[1][0], bounds[1][1], 300)
    X, Y = np.meshgrid(x_range, y_range)

    try:
        Z = target.evaluate([X, Y])
        if np.isscalar(Z):
            Z = np.full_like(X, Z, dtype=float)
    except Exception:
        Z = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Z[i, j] = target.evaluate([X[i, j], Y[i, j]])

    fig = go.Figure(data=[go.Contour(x=x_range, y=y_range, z=Z, colorscale='Viridis')])

    if projection_strategy is not None:
        mask = projection_strategy.get_feasibility_mask(X, Y)
        if not np.all(mask):
            shadow_z = (~mask).astype(float)
            fig.add_trace(go.Contour(
                x=x_range, y=y_range, z=shadow_z,
                showscale=False,
                colorscale=[[0, 'rgba(0,0,0,0)'], [1, 'rgba(0,0,0,0.2)']],
                hoverinfo='skip',
            ))

    if mode == 'trajectory':
        steps = history[:frame + 1]
        path = [step["population"][0] for step in steps]
        px = [p[0] for p in path]
        py = [p[1] for p in path]

        full_phases = [step.get("phase") for step in history]
        is_ravine = any(p is not None for p in full_phases)

        if is_ravine:
            # Classify on full history so block-end detection (xⁱ vs dⁱⱼ)
            # is correct even when the slider sits mid-descent. Same scheme
            # for single- and multi-step variants: the single-step variant
            # simply produces no 'd' frames (each descent block is length 1).
            kinds = _classify_ravine_frames(full_phases)[:frame + 1]

            fig.add_trace(go.Scatter(
                x=px, y=py, mode='lines',
                line=dict(color='rgba(220,150,160,0.4)', width=1),
                showlegend=False, hoverinfo='skip',
            ))

            def _trace(kind, color, size, symbol, border_color, border_width):
                xs = [px[i] for i, k in enumerate(kinds) if k == kind]
                ys = [py[i] for i, k in enumerate(kinds) if k == kind]
                if not xs:
                    return
                fig.add_trace(go.Scatter(
                    x=xs, y=ys, mode='markers',
                    marker=dict(color=color, size=size, symbol=symbol,
                                line=dict(color=border_color, width=border_width)),
                    showlegend=False, hoverinfo='skip',
                ))

            # Order matters — later traces draw on top.
            _trace('d', 'rgba(255,170,170,0.6)', 5, 'circle', 'rgba(0,0,0,0)', 0)
            _trace('v', 'cyan', 12, 'diamond', 'black', 1.2)
            _trace('x', '#ff2222', 12, 'circle', 'white', 1.5)
        else:
            fig.add_trace(go.Scatter(
                x=px, y=py, mode='lines+markers',
                marker=dict(color='red', size=8),
                line=dict(color='red', width=2),
            ))
    elif mode == 'swarm':
        current_pop = history[frame]["population"]
        fig.add_trace(go.Scatter(
            x=current_pop[:, 0], y=current_pop[:, 1],
            mode='markers', marker=dict(color='red', size=8),
        ))
    else:
        raise ValueError(f"Unknown contour mode: {mode!r}")

    fig.update_layout(title="Search Space", margin=dict(l=0, r=0, t=30, b=0), height=450)
    return fig


def build_convergence_figure(f_history, frame):
    iters = list(range(len(f_history)))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=iters, y=f_history, mode='lines',
                             line=dict(color='blue', width=2), name='f(x)'))
    fig.add_trace(go.Scatter(x=[frame], y=[f_history[frame]], mode='markers',
                             marker=dict(color='red', size=10), showlegend=False))
    fig.update_layout(title="Convergence Rate", xaxis_title="Iteration",
                      yaxis_title="Best f(x)", margin=dict(l=0, r=0, t=30, b=0), height=450)
    return fig


def build_history_dataframe(target, history, f_history, mode, frame):
    """
    mode: 'single' = include per-variable columns (GD/Newton).
          'population' = just iter + best f(x) (GA).
    """
    rows = []
    for i, step in enumerate(history[:frame + 1]):
        if mode == 'single':
            pt = step["population"][0]
            row = {"Iter": i}
            for j, var_name in enumerate(target.variables):
                row[var_name] = round(pt[j], 4)
            row["f(x)"] = round(f_history[i], 6)
            rows.append(row)
        elif mode == 'population':
            rows.append({"Iter": i, "Best f(x)": round(f_history[i], 6)})
        else:
            raise ValueError(f"Unknown history mode: {mode!r}")
    return pd.DataFrame(rows)
