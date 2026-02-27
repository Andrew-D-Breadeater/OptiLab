import numpy as np
from base import Optimizer
from scipy.optimize import minimize_scalar
from utils import logger

class GradientDescent(Optimizer):
    def __init__(self, target_function, **kwargs):
        super().__init__(target_function, **kwargs)
        self.learning_rate = kwargs.get('learning_rate', 0.01)
        self.decay_rate = kwargs.get('decay_rate', 1.0) # How fast the learning rate decays each iteration (1.0 means no decay)
        self.use_line_search = kwargs.get('use_line_search', False)
        self.use_exact_line_search = kwargs.get('use_exact_line_search', False)
        
        # Ravine parameters
        self.use_ravine = kwargs.get('use_ravine', False)
        self.ravine_step_size = kwargs.get('ravine_step_size', 0.5) # Represents 't'
        self.ravine_shift = kwargs.get('ravine_shift', 0.5) # The transverse shift amount
        self.prev_base_point = None 
        self.is_ravine_step = False
        self.initial_shift_done = False # Track if we created the second base point

    def _exact_line_search(self, x, direction):
        
        # Define the 1D function phi(alpha) = f(x + alpha * direction)
        def phi(alpha):
            return self.target.evaluate(x + alpha * direction)
            
        # Run 1D optimization to find the best alpha
        result = minimize_scalar(phi)
        return result.x

    def _backtracking_line_search(self, x, grad, direction, 
                                 alpha=1.0, beta=0.5, c=1e-4):
        """
        Finds alpha that satisfies the Armijo condition.
        alpha: initial step size
        beta: contraction factor (0 < beta < 1)
        c: sufficient decrease constant (0 < c < 1)
        """
        f_x = self.target.evaluate(x)
        # Precompute the dot product: grad^T * direction
        dot_product = np.dot(grad, direction)
        
        while True:
            # Calculate f(x + alpha * direction)
            f_next = self.target.evaluate(x + alpha * direction)
            
            # Check Armijo condition
            if f_next <= f_x + c * alpha * dot_product:
                return alpha
            
            # Reduce alpha
            alpha *= beta
            
            # Safety break to prevent infinite loop on flat regions
            if alpha < 1e-12:
                return alpha

    def step(self):
        # 1. Initialization: Create second base point via shift
        if self.use_ravine and not self.initial_shift_done:
            self.prev_base_point = self.current_x.copy()
            # Apply shift to all dimensions, or randomly to break symmetry
            shift_vector = np.ones_like(self.current_x) * self.ravine_shift
            shifted_x = self.current_x + shift_vector
            
            # Take one regular gradient step from the shifted point
            grad, is_subgrad = self.target.evaluate_gradient(shifted_x)
            self.used_subgradient = is_subgrad
            direction = -grad
            alpha = self._backtracking_line_search(shifted_x, grad, direction) if self.use_line_search else self.learning_rate
            
            self.initial_shift_done = True
            self.is_ravine_step = True # Next step will be a ravine step
            return shifted_x + alpha * direction

        # 2. Ravine Extrapolation Step
        if self.use_ravine and self.is_ravine_step and self.prev_base_point is not None:
            x_k = self.current_x
            x_prev = self.prev_base_point
            
            f_k = self.target.evaluate(x_k)
            f_prev = self.target.evaluate(x_prev)
            
            diff_vector = x_k - x_prev
            norm = np.linalg.norm(diff_vector)
            
            if norm == 0:
                self.is_ravine_step = False
                return x_k
            
            direction_normalized = diff_vector / norm
            sign_f = np.sign(f_k - f_prev)
            
            v_next = x_k - direction_normalized * self.ravine_step_size * sign_f
            
            self.prev_base_point = x_k.copy()
            self.is_ravine_step = False
            self.used_subgradient = False 
            
            return v_next

        # 3. Regular Gradient Step
        grad, is_subgrad = self.target.evaluate_gradient(self.current_x)
        self.used_subgradient = is_subgrad
        direction = -grad
        
        alpha = self._backtracking_line_search(self.current_x, grad, direction) if self.use_line_search else self.learning_rate
            
        new_x = self.current_x + alpha * direction
        
        if self.use_ravine:
            self.is_ravine_step = True
        
        self.learning_rate *= self.decay_rate # Apply decay to shrink step size over time
            
        return new_x