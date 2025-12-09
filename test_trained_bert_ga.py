"""
Test trained BERT mutation in full GA run and compare with baseline.
"""

import random
import torch
import time
import numpy as np
from problems.symbolic_regression import SymbolicRegression
from genetic_algorithm.tokenizer import SymbolicRegressionTokenizer
from genetic_algorithm.mutations.bert_mutation import SymbolicRegressionBERTMutation
from genetic_algorithm.crossovers.gp_crossover import PassthroughCrossover
from genetic_algorithm.ga import GeneticAlgorithm


def run_ga_with_bert(use_trained_model=True, num_generations=100, population_size=100, seed=42):
    """
    Run GA with BERT mutation.

    Args:
        use_trained_model: If True, load trained weights from models/
        num_generations: Number of generations to run
        population_size: Population size
        seed: Random seed

    Returns:
        dict with results
    """
    random.seed(seed)
    torch.manual_seed(seed)

    # Create problem - target: 2x^2 + 3x + 1
    problem = SymbolicRegression(
        target_function=lambda x: 2 * x**2 + 3 * x + 1,
        x_range=(-5, 5),
        num_points=50,
        max_tree_depth=5,
        penalty_coefficient=0.001
    )

    print(f"Target function: 2x^2 + 3x + 1")
    print(f"Population size: {population_size}")
    print(f"Generations: {num_generations}")
    print()

    # Create tokenizer
    tokenizer = SymbolicRegressionTokenizer(
        terminal_names=problem.get_terminal_set()
    )

    # Create BERT mutation
    mutation = SymbolicRegressionBERTMutation(
        tokenizer=tokenizer,
        masking_prob=0.2,
        temperature=1.0,
        epsilon_greedy=0.0,  # No exploration - use learned policy
        embedding_dim=64,
        num_heads=4,
        num_layers=2,
        chance=0.8,  # 80% mutation rate
        seed=seed
    )

    # Load trained model if requested
    if use_trained_model:
        try:
            mutation.load_model("models/bert_test_trained.pth")
            print("[+] Loaded trained BERT model from models/bert_test_trained.pth")
        except FileNotFoundError:
            print("[!] Trained model not found, using untrained model")
            use_trained_model = False
    else:
        print("[-] Using UNTRAINED BERT model")

    print()

    # Create GA
    crossover = PassthroughCrossover(seed=seed)
    ga = GeneticAlgorithm(
        crossover=crossover,
        mutation=mutation,
        fitness_function=problem.fitness_function,
        seed=seed
    )

    # Generate initial population
    initial_pop = problem.generate_population(population_size)
    ga.init_population(initial_pop)

    # Calculate initial stats manually (history not populated until start())
    initial_fitnesses = [problem.fitness_function(ind) for ind in initial_pop]
    initial_best = max(initial_fitnesses)
    initial_mean = sum(initial_fitnesses) / len(initial_fitnesses)

    print(f"Initial population stats:")
    print(f"  Best fitness: {initial_best:.6f}")
    print(f"  Mean fitness: {initial_mean:.6f}")
    print()

    # Run GA
    print(f"Running GA for {num_generations} generations...")
    start_time = time.time()

    best = ga.start(generations=num_generations, selector=20)

    elapsed = time.time() - start_time

    print()
    print(f"Completed in {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
    print()

    # Results
    best_fitness = problem.fitness_function(best)
    best_expr = problem.get_expression_string(best)

    print(f"Final Results:")
    print(f"  Best fitness: {best_fitness:.6f}")
    print(f"  Best expression: {best_expr}")
    print(f"  Best tree length: {len(best)}")
    print()

    # Analyze convergence
    history = ga.history
    initial_best = history['max'][0]
    final_best = history['max'][-1]
    improvement = final_best - initial_best

    # Find when we first reached near-optimal (within 1% of perfect)
    perfect_fitness = -0.001 * len(best)  # Just size penalty
    near_optimal_threshold = perfect_fitness * 1.01  # Within 1%

    gen_to_converge = num_generations
    for gen, fitness in enumerate(history['max']):
        if fitness >= near_optimal_threshold:
            gen_to_converge = gen
            break

    print(f"Evolution Progress:")
    print(f"  Initial best: {initial_best:.6f}")
    print(f"  Final best: {final_best:.6f}")
    print(f"  Improvement: {improvement:.6f}")
    print(f"  Generations to converge: {gen_to_converge}/{num_generations}")
    print()

    # Check if solution is correct
    test_points = [1, 2, 3, 4, 5]
    print(f"Solution verification (2x^2 + 3x + 1 test):")
    all_correct = True
    for x in test_points:
        expected = 2 * x**2 + 3 * x + 1
        actual = problem._evaluate_tree(best, {'x': x})
        error = abs(actual - expected)
        status = "[PASS]" if error < 0.1 else "[FAIL]"
        print(f"  x={x}: expected {expected}, got {actual:.2f} {status}")
        if error >= 0.1:
            all_correct = False

    print()
    if all_correct:
        print("[+] Solution is CORRECT!")
    else:
        print("[-] Solution is INCORRECT or suboptimal")

    return {
        'best_fitness': best_fitness,
        'best_expression': best_expr,
        'best_tree': best,
        'improvement': improvement,
        'gen_to_converge': gen_to_converge,
        'history': history,
        'elapsed_time': elapsed,
        'correct_solution': all_correct,
        'used_trained': use_trained_model
    }


def run_baseline_ga(num_generations=100, population_size=100, seed=42):
    """
    Run baseline GA with NO mutation (selection + elitism only).
    """
    random.seed(seed)

    problem = SymbolicRegression(
        target_function=lambda x: 2 * x**2 + 3 * x + 1,
        x_range=(-5, 5),
        num_points=50,
        max_tree_depth=5,
        penalty_coefficient=0.001
    )

    print(f"Target function: 2x^2 + 3x + 1")
    print(f"Population size: {population_size}")
    print(f"Generations: {num_generations}")
    print(f"Strategy: Selection + Elitism ONLY (no mutation)")
    print()

    # Create GA with no mutation
    from genetic_algorithm.mutations.base_mutation import BaseMutation

    class NoMutation(BaseMutation):
        def _mutate(self, offspring, **kwargs):
            return offspring

    crossover = PassthroughCrossover(seed=seed)
    mutation = NoMutation(chance=0.0, seed=seed)

    ga = GeneticAlgorithm(
        crossover=crossover,
        mutation=mutation,
        fitness_function=problem.fitness_function,
        seed=seed
    )

    initial_pop = problem.generate_population(population_size)
    ga.init_population(initial_pop)

    # Calculate initial stats manually (history not populated until start())
    initial_fitnesses = [problem.fitness_function(ind) for ind in initial_pop]
    initial_best = max(initial_fitnesses)
    initial_mean = sum(initial_fitnesses) / len(initial_fitnesses)

    print(f"Initial population stats:")
    print(f"  Best fitness: {initial_best:.6f}")
    print(f"  Mean fitness: {initial_mean:.6f}")
    print()

    print(f"Running baseline GA...")
    start_time = time.time()

    best = ga.start(generations=num_generations, selector=20)

    elapsed = time.time() - start_time

    best_fitness = problem.fitness_function(best)
    best_expr = problem.get_expression_string(best)

    print()
    print(f"Baseline Results:")
    print(f"  Best fitness: {best_fitness:.6f}")
    print(f"  Best expression: {best_expr}")
    print(f"  Improvement: {ga.history['max'][-1] - ga.history['max'][0]:.6f}")
    print()

    return {
        'best_fitness': best_fitness,
        'improvement': ga.history['max'][-1] - ga.history['max'][0],
        'history': ga.history
    }


def analyze_results(trained_results, untrained_results, baseline_results):
    """
    Analyze and compare results from all three runs.
    """
    print("=" * 80)
    print("COMPARATIVE ANALYSIS")
    print("=" * 80)
    print()

    print(f"{'Metric':<30} {'Baseline':<15} {'Untrained':<15} {'Trained':<15}")
    print("-" * 80)

    # Best fitness
    print(f"{'Best Fitness':<30} {baseline_results['best_fitness']:<15.6f} "
          f"{untrained_results['best_fitness']:<15.6f} {trained_results['best_fitness']:<15.6f}")

    # Improvement
    print(f"{'Fitness Improvement':<30} {baseline_results['improvement']:<15.6f} "
          f"{untrained_results['improvement']:<15.6f} {trained_results['improvement']:<15.6f}")

    # Convergence speed
    print(f"{'Generations to Converge':<30} {'N/A':<15} "
          f"{untrained_results['gen_to_converge']:<15} {trained_results['gen_to_converge']:<15}")

    # Solution correctness
    baseline_correct = "Unknown"
    print(f"{'Correct Solution Found':<30} {baseline_correct:<15} "
          f"{'Yes' if untrained_results['correct_solution'] else 'No':<15} "
          f"{'Yes' if trained_results['correct_solution'] else 'No':<15}")

    print()
    print("Best Expressions:")
    print(f"  Baseline:  {baseline_results.get('best_expression', 'N/A')}")
    print(f"  Untrained: {untrained_results['best_expression']}")
    print(f"  Trained:   {trained_results['best_expression']}")
    print()

    # Determine winner
    print("=" * 80)
    print("CONCLUSION")
    print("=" * 80)
    print()

    if trained_results['correct_solution'] and not untrained_results['correct_solution']:
        print("[+] TRAINED BERT found correct solution, UNTRAINED did not")
        print("  => Training is EFFECTIVE")
    elif trained_results['best_fitness'] > untrained_results['best_fitness'] + 0.01:
        print("[+] TRAINED BERT achieved significantly better fitness")
        print("  => Training is EFFECTIVE")
    elif trained_results['gen_to_converge'] < untrained_results['gen_to_converge'] * 0.8:
        print("[+] TRAINED BERT converged much faster")
        print("  => Training improves convergence speed")
    elif abs(trained_results['best_fitness'] - untrained_results['best_fitness']) < 0.01:
        print("[!] TRAINED and UNTRAINED have similar performance")
        print("  => May need more training or different problem")
    else:
        print("[-] UNTRAINED performed better than TRAINED")
        print("  => Possible issues:")
        print("    1. Model overtrained on training distribution")
        print("    2. Temperature/exploration settings need adjustment")
        print("    3. Training data doesn't match test scenario")

    print()

    # Check if BERT is better than baseline
    if trained_results['best_fitness'] > baseline_results['best_fitness'] + 0.01:
        print("[+] BERT (trained) is BETTER than baseline (no mutation)")
        print("  => Learned mutations are effective")
    elif untrained_results['best_fitness'] > baseline_results['best_fitness'] + 0.01:
        print("[+] BERT (untrained) is BETTER than baseline")
        print("  => Random mutations help, but training may not be optimal")
    else:
        print("[!] Baseline performs as well as BERT")
        print("  => PROBLEM: Mutations may not be helping or may be harmful")
        print("  => Possible issues:")
        print("    1. Mutation rate too high (destroying good solutions)")
        print("    2. Model learned to make harmful mutations")
        print("    3. Problem is too easy (good solutions in initial population)")

    print()


if __name__ == '__main__':
    print()
    print("#" * 80)
    print("# TRAINED BERT MUTATION - FULL GA TEST")
    print("# Testing with normal generation count and population size")
    print("#" * 80)
    print()

    NUM_GENS = 100
    POP_SIZE = 100
    SEED = 42

    print("=" * 80)
    print("TEST 1: BASELINE (No Mutation)")
    print("=" * 80)
    print()
    baseline_results = run_baseline_ga(
        num_generations=NUM_GENS,
        population_size=POP_SIZE,
        seed=SEED
    )

    print()
    print("=" * 80)
    print("TEST 2: UNTRAINED BERT")
    print("=" * 80)
    print()
    untrained_results = run_ga_with_bert(
        use_trained_model=False,
        num_generations=NUM_GENS,
        population_size=POP_SIZE,
        seed=SEED
    )

    print()
    print("=" * 80)
    print("TEST 3: TRAINED BERT")
    print("=" * 80)
    print()
    trained_results = run_ga_with_bert(
        use_trained_model=True,
        num_generations=NUM_GENS,
        population_size=POP_SIZE,
        seed=SEED
    )

    print()
    analyze_results(trained_results, untrained_results, baseline_results)

    print()
    print("#" * 80)
    print("# TEST COMPLETED")
    print("#" * 80)
    print()
