import numpy as np
from engine.optimizers.base import TraditionalOptimizer
from engine.strategies.gd_step import VanillaGradientStep


class GradientDescent(TraditionalOptimizer):
    """
    Gradient-descent optimizer. The actual per-iteration logic lives in the
    injected `step_strategy` (defaults to `VanillaGradientStep`).
    """
    def __init__(self, target_function, **kwargs):
        super().__init__(target_function, **kwargs)
        self.step_strategy = kwargs.get('step_strategy', VanillaGradientStep())
        self.used_subgradient = False

    def _get_history_state(self):
        state = super()._get_history_state()
        state["subgrad"] = self.used_subgradient
        return state

    def step(self):
        return self.step_strategy.step(self)
