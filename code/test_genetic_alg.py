import numpy as np
from engine.models import TargetFunction
from engine.initializers import RandomInitializer, HaltonInitializer
from engine.strategies.selection import ElitismSelection, TournamentSelection
from engine.strategies.crossover import UniformCrossover, NonUniformCrossover
from engine.strategies.mutation import RealCodedMutation
from engine.optimizers.population_based import GeneticAlgorithm

def run_ga_test(name, **kwargs):
    print(f"\n--- {name} ---")
    # Function: f(x, y) = x^2 + y^2 (Min at 0,0)
    target = TargetFunction("x**2 + y**2", bounds=[(-5, 5), (-5, 5)])
    
    ga = GeneticAlgorithm(
        target_function=target,
        population_size=50,
        **kwargs
    )
    
    # We no longer need to calculate 'best_x' or 'best_f' manually here
    # because the base class optimizer logs them and populates results.final_f
    ga.run(max_iter=50)

def test_ga_standard():
    run_ga_test(
        "Test 1: Standard GA (Random, Tournament, Uniform, Mutation)",
        initializer=RandomInitializer(),
        selection_strategy=TournamentSelection(tournament_size=3),
        crossover_strategy=UniformCrossover(),
        mutation_strategy=RealCodedMutation(sigma=0.2),
        phi_sel=0.2, phi_cross=0.6, phi_mut=0.2,
        stopping_criterion='stagnation',
        patience=5
    )

def test_ga_elitist():
    run_ga_test(
        "Test 2: Elitist GA (Halton, Elitism, Non-Uniform, Mutation)",
        initializer=HaltonInitializer(),
        selection_strategy=ElitismSelection(),
        crossover_strategy=NonUniformCrossover(),
        mutation_strategy=RealCodedMutation(sigma=0.1),
        phi_sel=0.2, phi_cross=0.6, phi_mut=0.2,
        stopping_criterion='degeneration',
        degeneration_tol=0.05
    )

if __name__ == "__main__":
    test_ga_standard()
    test_ga_elitist()