# code/main.py
import sys
from PyQt6.QtWidgets import QApplication
from gui.gui import OptimizerGUI

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = OptimizerGUI()
    window.show()
    sys.exit(app.exec())