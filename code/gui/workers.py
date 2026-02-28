from PyQt6.QtCore import QObject, pyqtSignal

class OptimizationWorker(QObject):
    progress = pyqtSignal(int)
    finished = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, optimizer_instance, max_iters, tol):
        super().__init__()
        self.optimizer = optimizer_instance
        self.max_iters = max_iters
        self.tol = tol

    def run(self):
        try:
            self.optimizer.run(max_iter=self.max_iters, tol=self.tol, callback=self.progress.emit)
            self.finished.emit(self.optimizer)
        except Exception as e:
            self.error.emit(str(e))