# OptiLab

A modular, headless optimization engine built in Python. It supports single-agent gradient methods and population-based evolutionary algorithms, paired with an interactive Streamlit visualization interface.

## Features
* **Gradient Methods**: Standard Gradient Descent, Backtracking/Exact Line Search, and Ravine Method.
* **Evolutionary Methods**: Genetic Algorithm with customizable strategies.
  * *Initializers*: Random, Halton (Quasi-Monte Carlo).
  * *Selection*: Tournament, Elitism, Roulette, Rank.
  * *Crossover*: Uniform, Non-Uniform.
  * *Mutation*: Real-Coded Gaussian.
* **Streamlit UI**: Interactive contour mapping, convergence tracking, and step-by-step animation using Plotly.

## Installation

1. Clone the repository and navigate to the root folder:
   ```bash
   cd OptiLab
   ```
2. Install the project in editable mode 
   ```bash
   pip install -e .
   ```

## Usage

### Running the Interactive UI
Navigate to the `code` directory and launch the Streamlit app:
```bash
cd code
streamlit run ui_interactive/app.py
```

### Running the Tests
To verify the engine mathematics and logic without the UI:
```bash
cd code
python tests/test_optimizer.py
python tests/test_genetic_alg.py
```