# Universal Optimizer GUI

A graphical interface for executing and visualizing numerical optimization methods. Developed to consolidate university laboratory projects into a single application.

## Overview
This tool evaluates user-defined mathematical functions and animates their convergence to a local or global minimum. It serves as a testbed for comparing the performance and behavior of different algorithms. 

## Features
* **Dynamic Input:** Parses string-based mathematical functions and boundaries.
* **Visualization:** Post-optimization animated plotting of the descent path and 1D convergence rate.
* **Algorithms:** Currently implements Gradient Descent (with backtracking/exact line search and ravine method options). I intend to implement several more traditional and evolutionary methods soon.
* **Stack:** Python, PyQt6, Matplotlib, NumPy, SymPy, SciPy.

## Installation
1. Clone the repository.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt