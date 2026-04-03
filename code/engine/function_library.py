# engine/function_library.py
import json
import os
from engine.utils import logger

FILE_PATH = "functions.json"

DEFAULT_FUNCTIONS = {
    "Sphere": {
        "expr": "x**2 + y**2",
        "bounds": "(-5, 5), (-5, 5)",
        "start_pos": "4.0, 4.0",
        "is_default": True
    },
    "Rosenbrock": {
        "expr": "(1 - x)**2 + 100 * (y - x**2)**2",
        "bounds": "(-5, 5), (-5, 5)",
        "start_pos": "-1.2, 1.0",
        "is_default": True
    },
    "Levy_13": {
        "expr": "sin(3*pi*x)**2 + (x-1)**2 * (1 + sin(3*pi*y)**2) + (y-1)**2 * (1 + sin(2*pi*y)**2)",
        "bounds": "(-10, 10), (-10, 10)",
        "start_pos": "0.0, 0.0",
        "is_default": True
    },
    "Eggholder": {
        "expr": "-(y+47)*sin(sqrt(abs(x/2 + y + 47))) - x*sin(sqrt(abs(x - (y + 47))))",
        "bounds": "(-512, 512), (-512, 512)",
        "start_pos": "0.0, 0.0",
        "is_default": True
    },
    "Himmelblau": {
        "expr": "(x**2 + y - 11)**2 + (x + y**2 - 7)**2",
        "bounds": "(-5, 5), (-5, 5)",
        "start_pos": "0.0, 0.0",
        "is_default": True
    },
    "Ackley": {
        "expr": "-20*exp(-0.2*sqrt(0.5*(x**2 + y**2))) - exp(0.5*(cos(2*pi*x) + cos(2*pi*y))) + exp(1) + 20",
        "bounds": "(-32, 32), (-32, 32)",
        "start_pos": "10.0, 10.0",
        "is_default": True
    }
}

class FunctionLibrary:
    def __init__(self, filepath=FILE_PATH):
        self.filepath = filepath
        self.functions = self._load()

    def _load(self):
        if not os.path.exists(self.filepath):
            logger.info("functions.json not found. Creating default library.")
            with open(self.filepath, 'w') as f:
                json.dump(DEFAULT_FUNCTIONS, f, indent=4)
            return DEFAULT_FUNCTIONS.copy()
        
        with open(self.filepath, 'r') as f:
            try:
                data = json.load(f)
                # Ensure defaults are always present even if file was edited
                for k, v in DEFAULT_FUNCTIONS.items():
                    if k not in data:
                        data[k] = v
                return data
            except Exception as e:
                logger.error(f"Failed to read functions.json: {e}")
                return DEFAULT_FUNCTIONS.copy()

    def save(self, name, expr, bounds, start_pos):
        self.functions[name] = {
            "expr": expr,
            "bounds": bounds,
            "start_pos": start_pos,
            "is_default": False
        }
        self._write()
        logger.info(f"Saved custom function preset: {name}")

    def delete(self, name):
        if name in self.functions and not self.functions[name].get("is_default", False):
            del self.functions[name]
            self._write()
            logger.info(f"Deleted custom function preset: {name}")

    def _write(self):
        with open(self.filepath, 'w') as f:
            json.dump(self.functions, f, indent=4)