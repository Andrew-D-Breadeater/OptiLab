import numpy as np
from engine.optimizers.base import TraditionalOptimizer


class ConjugateGradient(TraditionalOptimizer):
    """
    Conjugate gradient method for unconstrained / projection-constrained
    smooth minimization.

        h_0   = -∇f(x_0)
        h_k   = -∇f(x_k) + β_{k-1} h_{k-1}      (k ≥ 1, unless restart triggered)

    The β formula is the swappable knob — pass any concrete ``BetaStrategy``
    from ``engine.strategies.cg_beta`` (FR / PR / HS).

    Restart
    -------
    Every ``restart_every`` iterations we drop the conjugate term and take a
    plain anti-gradient step. Two sentinel values for the ``restart_every``
    kwarg:

    * ``None`` (default) → use the problem dimension ``n`` (textbook-recommended
      conservative period; valid for any quadratic since k ≤ n).
    * ``0`` → disable restart entirely. Step 0 is *still* a restart-equivalent
      because there is no ``prev_direction`` to combine with; this is required
      mathematically, not optional. The two cases are kept distinct so the
      "show why restart is necessary" lab demo can opt out cleanly.

    Note on the modulo guard: ``self.restart_every and step_idx % self.restart_every == 0``
    short-circuits when ``restart_every == 0`` — Python's ``and`` stops at the
    falsy ``0`` and the ``%`` operation never executes, so no ZeroDivisionError.
    """
    def __init__(self, target_function, *, beta_strategy,
                 restart_every: int | None = None, **kwargs):
        super().__init__(target_function, **kwargs)
        self.beta_strategy = beta_strategy
        n = len(target_function.variables)
        self.restart_every = n if restart_every is None else restart_every
        self._prev_grad: np.ndarray | None = None
        self._prev_direction: np.ndarray | None = None
        self._step_idx = 0

    def _is_restart_step(self) -> bool:
        if self._step_idx == 0:
            return True
        if not self.restart_every:
            return False
        return self._step_idx % self.restart_every == 0

    def _get_history_state(self):
        # Mark the originating dot: history[i] is the position from which
        # step_idx=i is taken, so the flag reflects the *upcoming* step.
        state = super()._get_history_state()
        state["restart"] = self._is_restart_step()
        return state

    def step(self):
        x = self.population[0]
        grad, _ = self.target.evaluate_gradient(x)
        self.current_grad = grad

        if self._is_restart_step():
            direction = -grad
        else:
            beta = self.beta_strategy.compute(grad, self._prev_grad, self._prev_direction)
            direction = -grad + beta * self._prev_direction

        alpha = self.get_alpha(x, grad, direction)
        new_x = self.projection_strategy.project(x + alpha * direction)

        self._prev_grad = grad
        self._prev_direction = direction
        self._step_idx += 1

        return np.atleast_2d(new_x)
