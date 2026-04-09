import numpy as np
from scipy.optimize import minimize_scalar
from engine.optimizers.base import Optimizer
from engine.utils import logger

import numpy as np
from engine.optimizers.base import TraditionalOptimizer

class GradientDescent(TraditionalOptimizer):
    def __init__(self, target_function, **kwargs):
        super().__init__(target_function, **kwargs)
        self.decay_rate = kwargs.get('decay_rate', 1.0)
        
        self.use_ravine = kwargs.get('use_ravine', False)
        self.ravine_step_size = kwargs.get('ravine_step_size', 0.5)
        self.ravine_shift = kwargs.get('ravine_shift', 0.5)
        self.prev_base_point = None 
        self.is_ravine_step = False
        self.initial_shift_done = False
        
        self.used_subgradient = False

    def _get_history_state(self):
        state = super()._get_history_state()
        state["subgrad"] = self.used_subgradient
        return state

    def step(self):
        current_x = self.population[0] # Extract the single point for GD
        
        # 1. Initialization: Create second base point via shift
        if self.use_ravine and not self.initial_shift_done:
            self.prev_base_point = current_x.copy()
            shift_vector = np.ones_like(current_x) * self.ravine_shift
            shifted_x = current_x + shift_vector
            
            grad, is_subgrad = self.target.evaluate_gradient(shifted_x)
            self.used_subgradient = is_subgrad
            self.current_grad = grad
            direction = -grad
            
            alpha = self.get_alpha(shifted_x, grad, direction)
            
            self.initial_shift_done = True
            self.is_ravine_step = True
            
            new_x = self.projection_strategy.project(shifted_x + alpha * direction)
            return np.atleast_2d(new_x)

        # 2. Ravine Extrapolation Step
        if self.use_ravine and self.is_ravine_step and self.prev_base_point is not None:
            x_k = current_x
            x_prev = self.prev_base_point
            
            f_k = self.target.evaluate(x_k)
            f_prev = self.target.evaluate(x_prev)
            
            diff_vector = x_k - x_prev
            norm = np.linalg.norm(diff_vector)
            
            if norm == 0:
                self.is_ravine_step = False
                return np.atleast_2d(x_k)
            
            direction_normalized = diff_vector / norm
            sign_f = np.sign(f_k - f_prev)
            
            v_next = x_k - direction_normalized * self.ravine_step_size * sign_f
            
            self.prev_base_point = x_k.copy()
            self.is_ravine_step = False
            self.used_subgradient = False 
            
            new_x = self.projection_strategy.project(v_next)
            return np.atleast_2d(new_x)

        # 3. Regular Gradient Step
        grad, is_subgrad = self.target.evaluate_gradient(current_x)
        self.used_subgradient = is_subgrad
        self.current_grad = grad
        direction = -grad
        
        alpha = self.get_alpha(current_x, grad, direction)
            
        new_x = self.projection_strategy.project(current_x + alpha * direction)
        
        if self.use_ravine:
            self.is_ravine_step = True
        
        self.learning_rate *= self.decay_rate
            
        return np.atleast_2d(new_x)