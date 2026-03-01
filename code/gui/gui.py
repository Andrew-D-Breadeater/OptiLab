import logging
import ast
import numpy as np
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, 
                             QPushButton, QLabel, QLineEdit, QComboBox, 
                             QFrame, QStackedWidget, QProgressBar, 
                             QSizePolicy, QCheckBox, QMainWindow)
from PyQt6.QtCore import Qt, QTimer, QThread

from gui.plotter import MplCanvas
from gui.workers import OptimizationWorker
from engine.utils import check_convexity
from engine.models import TargetFunction
from engine.optimizers.gradient_methods import GradientDescent

gui_logger = logging.getLogger("gui")
gui_logger.setLevel(logging.INFO)
handler = logging.FileHandler("gui.log")
handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] GUI: %(message)s'))
gui_logger.addHandler(handler)

class OptimizerGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Optimization Engine")
        self.resize(1200, 700)

        # Central Layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # --- LEFT PANEL: Method Specific Options ---
        self.left_panel = QFrame()
        self.left_panel.setFrameShape(QFrame.Shape.StyledPanel)
        self.left_layout = QVBoxLayout(self.left_panel)
        self.left_layout.addWidget(QLabel("<b>Method Specific Options</b>"))
        self.left_layout.addStretch() # Push items to top
        
        # Panel Toggle Button
        self.toggle_btn = QPushButton("<")
        self.toggle_btn.setFixedWidth(20)
        self.toggle_btn.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        self.toggle_btn.clicked.connect(self.toggle_left_panel)

        main_layout.addWidget(self.left_panel)
        main_layout.addWidget(self.toggle_btn)

        # --- RIGHT PANEL ---
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        main_layout.addWidget(right_panel, stretch=1)

        # 1. Top Section: The State Board (Inputs -> Progress -> Results)
        self.state_board = QStackedWidget()
        self.state_board.setFixedHeight(120)
        right_layout.addWidget(self.state_board)

        self._init_input_page()
        self._init_progress_page()
        self._init_results_page()
        
        # function convexity status
        graph_header_layout = QHBoxLayout()
        self.convex_label = QLabel("<b>Convexity:</b> Unknown")
        graph_header_layout.addWidget(self.convex_label)
        right_layout.addLayout(graph_header_layout)

        # 2. Bottom Section: Graphs
        graph_layout = QHBoxLayout()
        self.canvas_func = MplCanvas(self)
        self.canvas_conv = MplCanvas(self)
        
        self.canvas_func.axes.set_title("Function Graph")
        self.canvas_conv.axes.set_title("Convergence Rate")
        
        graph_layout.addWidget(self.canvas_func)
        graph_layout.addWidget(self.canvas_conv)
        right_layout.addLayout(graph_layout)
        
        #3 Debounce timer setup
        self.plot_timer = QTimer()
        self.plot_timer.setSingleShot(True)
        self.plot_timer.timeout.connect(self._trigger_evaluation)
        
        #4 Initial left panel setup
        self._update_left_panel()
        
        #5 Initial Plot trigger
        self._trigger_evaluation()
        
        gui_logger.info("Optimizer GUI initialized and ready.")

    def _init_input_page(self):
        page = QWidget()
        layout = QGridLayout(page)
        
        # Row 0
        layout.addWidget(QLabel("Target Function f(x):"), 0, 0)
        self.func_input = QLineEdit("x**2 + y**2")
        self.func_input.textChanged.connect(self._on_typing) # Connected to the final instance
        layout.addWidget(self.func_input, 0, 1)

        layout.addWidget(QLabel("Starting point (x, y):"), 0, 2)
        self.start_input = QLineEdit("5.0, 5.0")
        layout.addWidget(self.start_input, 0, 3)

        # Row 1
        layout.addWidget(QLabel("Optimization Method:"), 1, 0)
        self.method_combo = QComboBox()
        self.method_combo.addItems(["Gradient Descent"]) # Only one core engine for now
        self.method_combo.currentIndexChanged.connect(self._update_left_panel)
        layout.addWidget(self.method_combo, 1, 1)

        layout.addWidget(QLabel("Bounds:"), 1, 2)
        self.bounds_input = QLineEdit("(-10, 10), (-10, 10)")
        self.bounds_input.textChanged.connect(self._on_typing)
        layout.addWidget(self.bounds_input, 1, 3)

        # Row 2
        self.start_btn = QPushButton("Start")
        self.start_btn.clicked.connect(self._on_start_clicked)
        layout.addWidget(self.start_btn, 2, 3)
        
        self.state_board.addWidget(page) # Index 0
        gui_logger.info("Input page initialized and ready.") 

    def _init_progress_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        
        # Fixed alignment assignment
        status_lbl = QLabel("<b>Optimizing...</b>")
        status_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(status_lbl)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0) 
        layout.addWidget(self.progress_bar)
        
        self.cancel_btn = QPushButton("Cancel / Force Show Results")
        self.cancel_btn.clicked.connect(self._on_cancel_clicked) 
        layout.addWidget(self.cancel_btn)
        
        self.state_board.addWidget(page) 
        gui_logger.info("Progress page initialized and ready.")

    def _init_results_page(self):
        page = QWidget()
        layout = QGridLayout(page)
        
        self.res_title = QLabel("<b>Optimization Complete</b>")
        self.res_details = QLabel("Converged in N steps.\nFinal x: [...]\nFinal f(x): ...")
        
        self.reset_btn = QPushButton("New Optimization")
        self.reset_btn.clicked.connect(self._on_reset_clicked) 
        
        layout.addWidget(self.res_title, 0, 0)
        layout.addWidget(self.res_details, 1, 0)
        layout.addWidget(self.reset_btn, 0, 1, 2, 1, Qt.AlignmentFlag.AlignRight)
        
        self.state_board.addWidget(page)
        gui_logger.info("Results page initialized and ready.") 

    def toggle_left_panel(self):
        if self.left_panel.isVisible():
            gui_logger.debug("Hiding method specific options panel.")
            self.left_panel.hide()
            self.toggle_btn.setText(">")
        else:
            gui_logger.debug("Showing method specific options panel.")
            self.left_panel.show()
            self.toggle_btn.setText("<")
            
    def _update_left_panel(self):
        # Clear existing layout (except the title at index 0)
        for i in reversed(range(1, self.left_layout.count())):
            item = self.left_layout.itemAt(i)
            if item is not None:
                widget = item.widget()
                if widget is not None:
                    widget.setParent(None)
                elif item.spacerItem() is not None:
                    self.left_layout.removeItem(item)

        method = self.method_combo.currentText()
        gui_logger.info(f"Updating left panel parameters for: {method}")

        if method == "Gradient Descent":
            # Core Engine Parameters
            self.left_layout.addWidget(QLabel("Learning Rate:"))
            self.lr_input = QLineEdit("0.01")
            self.left_layout.addWidget(self.lr_input)

            self.left_layout.addWidget(QLabel("Max Iterations:"))
            self.iter_input = QLineEdit("1000")
            self.left_layout.addWidget(self.iter_input)

            self.left_layout.addWidget(QLabel("Tolerance:"))
            self.tol_input = QLineEdit("1e-6")
            self.left_layout.addWidget(self.tol_input)

            # Line Search Toggles
            self.bt_checkbox = QCheckBox("Backtracking Line Search")
            self.left_layout.addWidget(self.bt_checkbox)

            self.exact_checkbox = QCheckBox("Exact Line Search")
            self.left_layout.addWidget(self.exact_checkbox)

            # Ravine Toggles
            self.ravine_checkbox = QCheckBox("Use Ravine Method")
            self.left_layout.addWidget(self.ravine_checkbox)

            self.left_layout.addWidget(QLabel("Ravine Step Size:"))
            self.ravine_step = QLineEdit("0.5")
            self.left_layout.addWidget(self.ravine_step)

        self.left_layout.addStretch()
    
    def _on_start_clicked(self):
        gui_logger.info("Parsing inputs and initializing optimization thread.")
        try:
            expr = self.func_input.text()
            bounds = list(ast.literal_eval(self.bounds_input.text()))
            start_point = list(ast.literal_eval(self.start_input.text()))
            
            target = TargetFunction(expr, bounds=bounds)
            method = self.method_combo.currentText()
            
            if method == "Gradient Descent":
                max_iters = int(self.iter_input.text())
                tol = float(self.tol_input.text())
                
                kwargs = {
                    'start_pos': np.array(start_point, dtype=float),
                    'learning_rate': float(self.lr_input.text()),
                    'use_line_search': self.bt_checkbox.isChecked(),
                    'use_exact_line_search': self.exact_checkbox.isChecked(),
                    'use_ravine': self.ravine_checkbox.isChecked()
                }
                
                if self.ravine_checkbox.isChecked():
                    kwargs['ravine_step_size'] = float(self.ravine_step.text())
                    
                optimizer = GradientDescent(target, **kwargs)
            else:
                raise ValueError(f"Method '{method}' is not implemented yet.")
            
            # Setup UI for progress
            self.progress_bar.setMaximum(max_iters)
            self.progress_bar.setValue(0)
            self.state_board.setCurrentIndex(1)
            
            # Threading Setup (renamed variables to avoid shadowing)
            self.optim_thread = QThread()
            self.optim_worker = OptimizationWorker(optimizer, max_iters, tol)
            self.optim_worker.moveToThread(self.optim_thread)
            
            # Connect Signals
            self.optim_thread.started.connect(self.optim_worker.run)
            self.optim_worker.progress.connect(self._update_progress)
            self.optim_worker.finished.connect(self._on_optimization_finished)
            self.optim_worker.error.connect(self._on_optimization_error)
            
            self.optim_worker.finished.connect(self.optim_thread.quit)
            self.optim_worker.finished.connect(self.optim_worker.deleteLater)
            self.optim_thread.finished.connect(self.optim_thread.deleteLater)
            
            self.optim_thread.start()

        except Exception as e:
            gui_logger.error(f"Initialization failed: {e}")
            self.res_title.setText("<b>Initialization Failed</b>")
            self.res_details.setText(str(e))
            self.state_board.setCurrentIndex(2)

    def _update_progress(self, iteration):
        self.progress_bar.setValue(iteration)
        
    def _on_optimization_finished(self, final_optimizer_state):
        gui_logger.info("Optimization thread completed successfully.")
        
        res = final_optimizer_state.results
        target = final_optimizer_state.target
        self.optim_history = res.history
        
        # 1. Update UI Text
        x_str = ", ".join([f"{val:.4f}" for val in res.final_x])
        details = (
            f"Converged: {res.converged} in {res.iterations} steps\n"
            f"Final x: [{x_str}]\n"
            f"Final f(x): {res.final_f:.6f}\n"
            f"Execution Time: {res.execution_time:.4f} sec"
        )
        self.res_title.setText("<b>Optimization Complete</b>")
        self.res_details.setText(details)
        self.state_board.setCurrentIndex(2)
        
        # 2. Setup Data & Base Graphs
        self.f_vals = [target.evaluate(step["x"]) for step in self.optim_history]
        self.canvas_conv.draw_convergence_base(self.f_vals)
        self.canvas_func.init_contour_animation()
        
        # 3. Setup Synchronized Animation Timer
        self.anim_frame = 0
        self.anim_timer = QTimer()
        self.anim_timer.timeout.connect(self._play_animation_frame)
        self.anim_timer.start(30) # 30ms per frame

    def _play_animation_frame(self):
        # Stop condition
        if self.anim_frame >= len(self.optim_history):
            self.anim_timer.stop()
            return
            
        # Extract coordinates up to current frame
        current_path = [step["x"] for step in self.optim_history[:self.anim_frame + 1]]
        path_x = [pt[0] for pt in current_path]
        path_y = [pt[1] for pt in current_path]
        
        # Update both graphs
        self.canvas_func.update_contour_frame(path_x, path_y)
        self.canvas_conv.update_convergence_frame(self.anim_frame, self.f_vals[self.anim_frame])
        
        # Frame jumping to prevent UI lock on massive histories (max 100 frames total)
        step_size = max(1, len(self.optim_history) // 100)
        
        # Ensure we don't skip the final frame
        if self.anim_frame < len(self.optim_history) - 1 and self.anim_frame + step_size >= len(self.optim_history):
            self.anim_frame = len(self.optim_history) - 1
        else:
            self.anim_frame += step_size
        
    def _on_optimization_error(self, err_msg):
        gui_logger.error(f"Optimization failed during execution: {err_msg}")
        self.res_title.setText("<b>Optimization Failed</b>")
        self.res_details.setText(err_msg)
        self.state_board.setCurrentIndex(2)

    def _on_cancel_clicked(self):
        gui_logger.info("Optimization cancelled/forced to results by user.")
        self.state_board.setCurrentIndex(2)

    def _on_reset_clicked(self):
        gui_logger.info("Resetting UI for new optimization.")
        self.state_board.setCurrentIndex(0)
        
    def _on_typing(self, text=""):
        gui_logger.debug(f"User is typing a new function: {text}")
        self.plot_timer.start(600) # Restart the timer on every keystroke, so it only triggers after 600ms of inactivity.

    def _trigger_evaluation(self):
        expr = self.func_input.text()
        bounds_str = self.bounds_input.text()
        
        try:
            # 1. Validate Bounds
            try:
                bounds = ast.literal_eval(bounds_str)
                if not isinstance(bounds, (list, tuple)) or not all(isinstance(b, (list, tuple)) for b in bounds):
                    raise ValueError("Bounds must be a sequence of coordinate pairs.")
            except Exception as e:
                raise ValueError(f"Invalid Bounds Format")

            # 2. Validate Expression
            try:
                target = TargetFunction(expr, bounds=list(bounds))
            except Exception as e:
                raise ValueError(f"Invalid Math Expression")

            gui_logger.debug(f"Evaluating new TargetFunction: {expr}, bounds: {bounds}")
            
            self.canvas_func.draw_contour(target)
            
            is_convex, counter_point = check_convexity(expr, ['x', 'y'], bounds)
            
            if is_convex is True:
                self.convex_label.setText("<b>Convexity:</b> <span style='color:green;'>PSD Confirmed</span>")
            elif is_convex is False:
                self.convex_label.setText(f"<b>Convexity:</b> <span style='color:red;'>Failed at {counter_point}</span>")
            else:
                self.convex_label.setText("<b>Convexity:</b> Error / Unknown")

        except ValueError as e:
            # Catches both custom ValueErrors raised above
            error_type = str(e)
            gui_logger.debug(f"Evaluation deferred: {error_type}")
            
            self.canvas_func.axes.clear()
            self.canvas_func.axes.set_title(f"Waiting: {error_type}...")
            self.canvas_func.draw()
            self.convex_label.setText("<b>Convexity:</b> Waiting for valid input...")