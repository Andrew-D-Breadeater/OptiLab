import logging
import numpy as np
import sympy as sp
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

logger = logging.getLogger("gui.plotter")

class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)

    def draw_contour(self, target_function):
        self.axes.clear()
        try:
            bounds = getattr(target_function, 'bounds', None)
            if not bounds or len(bounds) != 2:
                raise ValueError("Contour plots require exactly 2 dimensions.")

            # Create meshgrid
            x_val = np.linspace(bounds[0][0], bounds[0][1], 100)
            y_val = np.linspace(bounds[1][0], bounds[1][1], 100)
            X, Y = np.meshgrid(x_val, y_val)
            
            # Evaluate using TargetFunction's built-in lambdified method
            Z = target_function.evaluate([X, Y])
            
            # Matplotlib contour requires a 2D array. Fix scalars (e.g., f(x) = 5)
            if np.isscalar(Z):
                Z = np.full_like(X, Z, dtype=float)
                
            self.axes.contour(X, Y, Z, levels=50, cmap='viridis')
            
            title = getattr(target_function, 'expression_str', 'Callable Function')
            self.axes.set_title(f"Contour: {title}")
            logger.info(f"Successfully plotted contour for: {title}")
            
        except Exception as e:
            self.axes.set_title("Waiting for valid 2D expression/bounds...")
            logger.debug(f"Plotting deferred. Reason: {e}")
        
        self.draw()
        
    def init_contour_animation(self):
        # Remove existing animation lines to prevent ghost paths
        if hasattr(self, 'path_line') and self.path_line in self.axes.lines:
            self.path_line.remove()
        if hasattr(self, 'head_dot') and self.head_dot in self.axes.lines:
            self.head_dot.remove()

        # Create fresh empty line objects
        self.path_line, = self.axes.plot([], [], 'r-', linewidth=1.5, alpha=0.7)
        self.head_dot, = self.axes.plot([], [], 'ro', markersize=6)
        self.draw() # Force a redraw to wipe the old path instantly

    def draw_convergence_base(self, f_history):
        self.axes.clear()
        iterations = range(len(f_history))
        # Draw the static background curve
        self.axes.plot(iterations, f_history, 'b-', linewidth=2, alpha=0.3)
        # Create empty moving dot
        self.conv_dot, = self.axes.plot([], [], 'ro', markersize=8) 
        
        self.axes.set_title("Convergence Rate")
        self.axes.set_xlabel("Iteration")
        self.axes.set_ylabel("$f(x)$")
        self.axes.grid(True, linestyle='--', alpha=0.6)
        self.draw()

    def update_contour_frame(self, path_x, path_y):
        self.path_line.set_data(path_x, path_y)
        if path_x and path_y:
            self.head_dot.set_data([path_x[-1]], [path_y[-1]])
        self.draw()

    def update_convergence_frame(self, iteration, f_val):
        self.conv_dot.set_data([iteration], [f_val])
        self.draw()