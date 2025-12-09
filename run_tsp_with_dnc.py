"""
TSP Genetic Algorithm with DNC/LSTM Crossover
Generates GIF visualization, plots, and comprehensive metrics
"""

import argparse
import random
import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from datetime import datetime
import json
import imageio
from typing import List, Tuple
from tqdm import tqdm

from genetic_algorithm.ga import GeneticAlgorithm
from genetic_algorithm.rl_trainer import RLTrainer
from genetic_algorithm.crossovers.dnc_crossover import DNCrossover
from genetic_algorithm.mutations.generic_mutations import SwapMutation
from problems.tsp import TSPProblem


class TSPWithCorrectness(TSPProblem):
    """
    Enhanced TSP problem with correctness checking in fitness function.
    This penalizes invalid permutations (repetitions, missing cities).
    """

    def __init__(self, num_cities: int = 20, seed: int = None, penalty_weight: float = 1000.0,
                 use_soft_penalty: bool = True):
        super().__init__(num_cities, seed)
        self.penalty_weight = penalty_weight
        self.use_soft_penalty = use_soft_penalty
        # Estimate of worst possible tour distance for scaling
        self.max_distance_estimate = num_cities * 150  # Approximate worst case

    def fitness_function(self, individual: list) -> float:
        """
        Calculate fitness with correctness checking.

        Fitness = -distance - penalty

        For soft penalty mode:
        - Still evaluates tour distance even if invalid
        - Adds graduated penalty based on violations
        - This gives better learning signal to DNC

        Penalty is applied for:
        - Wrong length
        - Duplicate cities
        - Missing cities
        - Invalid city indices
        """
        penalty = 0.0

        # Check 1: Length correctness
        if len(individual) != self.num_cities:
            penalty += self.penalty_weight * abs(len(individual) - self.num_cities)

        # Check 2: All cities in valid range
        invalid_cities = sum(1 for city in individual if city < 0 or city >= self.num_cities)
        penalty += self.penalty_weight * invalid_cities

        # Check 3: Check for duplicates
        unique_cities = len(set(individual))
        duplicates = len(individual) - unique_cities
        penalty += self.penalty_weight * duplicates

        # Check 4: Check for missing cities (should be a permutation of 0...n-1)
        expected_cities = set(range(self.num_cities))
        actual_cities = set(c for c in individual if 0 <= c < self.num_cities)
        missing_cities = len(expected_cities - actual_cities)
        penalty += self.penalty_weight * missing_cities

        # Calculate tour distance
        if self.use_soft_penalty and len(individual) > 0 and duplicates == 0:
            # Even with some violations, try to evaluate tour distance
            # This gives DNC better gradient signal
            try:
                total_distance = 0.0
                for i in range(min(len(individual) - 1, self.num_cities)):
                    if 0 <= individual[i] < self.num_cities and 0 <= individual[i + 1] < self.num_cities:
                        total_distance += self._distance(individual[i], individual[i + 1])
                if len(individual) > 0 and 0 <= individual[-1] < self.num_cities and 0 <= individual[0] < self.num_cities:
                    total_distance += self._distance(individual[-1], individual[0])

                # Normalize distance to similar scale as penalty
                normalized_distance = total_distance

                return -(normalized_distance + penalty)
            except:
                return -penalty
        elif penalty > 0:
            # Hard penalty mode or severe violations
            return -self.max_distance_estimate - penalty
        else:
            # Valid individual
            total_distance = 0.0
            for i in range(len(individual) - 1):
                total_distance += self._distance(individual[i], individual[i + 1])
            total_distance += self._distance(individual[-1], individual[0])
            return -total_distance

    def is_valid_individual(self, individual: list) -> bool:
        """Check if individual is a valid permutation"""
        if len(individual) != self.num_cities:
            return False
        if any(city < 0 or city >= self.num_cities for city in individual):
            return False
        if len(set(individual)) != self.num_cities:
            return False
        return True

    def get_tour_distance(self, individual: list) -> float:
        """Get tour distance (without penalty)"""
        if not self.is_valid_individual(individual):
            return float('inf')

        total_distance = 0.0
        for i in range(len(individual) - 1):
            total_distance += self._distance(individual[i], individual[i + 1])
        total_distance += self._distance(individual[-1], individual[0])
        return total_distance


def visualize_tsp_tour(
    cities: List[Tuple[float, float]],
    tour: list,
    ax,
    title: str = "TSP Tour"
):
    """
    Visualize a TSP tour on given axes.

    Args:
        cities: List of (x, y) city coordinates
        tour: Order of cities to visit
        ax: Matplotlib axes object
        title: Plot title
    """
    ax.clear()

    # Plot cities
    city_x = [cities[i][0] for i in range(len(cities))]
    city_y = [cities[i][1] for i in range(len(cities))]
    ax.scatter(city_x, city_y, c='red', s=100, zorder=5, label='Cities')

    # Plot tour
    if len(tour) > 0:
        tour_x = [cities[tour[i]][0] for i in range(len(tour))]
        tour_y = [cities[tour[i]][1] for i in range(len(tour))]
        # Close the tour
        tour_x.append(cities[tour[0]][0])
        tour_y.append(cities[tour[0]][1])

        ax.plot(tour_x, tour_y, 'b-', alpha=0.6, linewidth=2, label='Tour')

    # Annotate cities with indices
    for i, (x, y) in enumerate(cities):
        ax.annotate(str(i), (x, y), fontsize=8, ha='center', va='center',
                   bbox=dict(boxstyle='circle', facecolor='white', alpha=0.7))

    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-5, 105)
    ax.set_ylim(-5, 105)


def create_tsp_gif(
    cities: List[Tuple[float, float]],
    tour_history: list,
    distance_history: list,
    generation_history: list,
    save_path: str,
    fps: int = 2
):
    """
    Create animated GIF showing TSP optimization progress.

    Args:
        cities: List of city coordinates
        tour_history: List of best tours at each snapshot
        distance_history: List of best distances
        generation_history: List of generation numbers
        save_path: Path to save GIF
        fps: Frames per second
    """
    print(f"\nGenerating TSP optimization GIF...")
    frames = []

    for idx, (tour, distance, gen) in enumerate(zip(tour_history, distance_history, generation_history)):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Plot tour
        visualize_tsp_tour(
            cities,
            tour,
            ax1,
            title=f'Best Tour at Generation {gen}\nDistance: {distance:.2f}'
        )

        # Plot distance history
        ax2.plot(generation_history[:idx+1], distance_history[:idx+1],
                'g-', linewidth=2, marker='o', markersize=4)
        ax2.set_xlabel('Generation')
        ax2.set_ylabel('Tour Distance')
        ax2.set_title('Optimization Progress')
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, max(generation_history))
        if len(distance_history) > 1:
            y_min = min(distance_history) * 0.95
            y_max = max(distance_history) * 1.05
            ax2.set_ylim(y_min, y_max)

        plt.tight_layout()

        # Save frame to buffer
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.buffer_rgba(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        # Convert RGBA to RGB
        image = image[:, :, :3]
        frames.append(image)

        plt.close(fig)

    # Save as GIF
    imageio.mimsave(save_path, frames, fps=fps, loop=0)
    print(f"[OK] GIF saved to {save_path}")


def plot_comprehensive_results(
    ga_history: dict,
    problem: TSPWithCorrectness,
    best_tour: list,
    metrics: dict,
    save_dir: Path
):
    """
    Generate comprehensive result plots.

    Args:
        ga_history: GA history with mean and max fitness
        problem: TSP problem instance
        best_tour: Best tour found
        metrics: Dictionary of additional metrics
        save_dir: Directory to save plots
    """
    print(f"\nGenerating comprehensive plots...")

    # Plot 1: Fitness evolution
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    generations = range(len(ga_history['mean']))

    # Subplot 1: Mean fitness
    axes[0, 0].plot(generations, ga_history['mean'], 'b-', alpha=0.7, linewidth=2)
    axes[0, 0].set_xlabel('Generation')
    axes[0, 0].set_ylabel('Mean Fitness')
    axes[0, 0].set_title('Mean Population Fitness Evolution')
    axes[0, 0].grid(True, alpha=0.3)

    # Subplot 2: Best fitness
    axes[0, 1].plot(generations, ga_history['max'], 'g-', alpha=0.7, linewidth=2)
    axes[0, 1].set_xlabel('Generation')
    axes[0, 1].set_ylabel('Best Fitness')
    axes[0, 1].set_title('Best Fitness Evolution')
    axes[0, 1].grid(True, alpha=0.3)

    # Subplot 3: Best tour distances (convert fitness to distance)
    distances = [-f for f in ga_history['max']]
    axes[1, 0].plot(generations, distances, 'r-', alpha=0.7, linewidth=2)
    axes[1, 0].set_xlabel('Generation')
    axes[1, 0].set_ylabel('Tour Distance')
    axes[1, 0].set_title('Best Tour Distance Over Time')
    axes[1, 0].grid(True, alpha=0.3)

    # Subplot 4: Best tour visualization
    visualize_tsp_tour(
        problem.cities,
        best_tour,
        axes[1, 1],
        f'Final Best Tour\nDistance: {problem.get_tour_distance(best_tour):.2f}'
    )

    plt.tight_layout()
    save_path = save_dir / 'fitness_evolution.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[OK] Fitness evolution plot saved to {save_path}")
    plt.close()

    # Plot 2: Metrics summary
    if 'validity_rate' in metrics:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Validity rate over time
        axes[0].plot(metrics['validity_rate'], 'purple', linewidth=2)
        axes[0].set_xlabel('Generation')
        axes[0].set_ylabel('Validity Rate (%)')
        axes[0].set_title('Population Validity Rate Over Time')
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim(-5, 105)

        # Diversity over time (if available)
        if 'diversity' in metrics:
            axes[1].plot(metrics['diversity'], 'orange', linewidth=2)
            axes[1].set_xlabel('Generation')
            axes[1].set_ylabel('Population Diversity')
            axes[1].set_title('Population Diversity Over Time')
            axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = save_dir / 'population_metrics.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[OK] Population metrics plot saved to {save_path}")
        plt.close()


def calculate_population_diversity(population: list) -> float:
    """
    Calculate population diversity as average pairwise distance between individuals.
    """
    if len(population) <= 1:
        return 0.0

    total_distance = 0.0
    count = 0

    for i in range(len(population)):
        for j in range(i + 1, len(population)):
            # Hamming distance between permutations
            distance = sum(1 for k in range(len(population[i]))
                         if population[i][k] != population[j][k])
            total_distance += distance
            count += 1

    return total_distance / count if count > 0 else 0.0


def run_tsp_with_dnc(
    num_cities: int = 20,
    population_size: int = 100,
    generations: int = 200,
    selector: int = 20,
    mutation_rate: float = 0.3,
    hidden_size: int = 128,
    num_layers: int = 2,
    train_dnc: bool = True,
    train_mode: str = 'online',
    pretrain_episodes: int = 100,
    online_interval: int = 10,
    online_episodes: int = 10,
    warmup_gens: int = 20,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    penalty_weight: float = 1000.0,
    use_soft_penalty: bool = True,
    reward_type: str = "absolute",
    temperature: float = 1.0,
    seed: int = 42,
    output_dir: str = 'tsp_results',
    gif_fps: int = 2,
    snapshot_interval: int = 5
):
    """
    Run TSP GA with DNC/LSTM crossover, visualization, and metrics.

    Args:
        num_cities: Number of cities in TSP
        population_size: GA population size
        generations: Number of GA generations
        selector: Number of worst individuals to replace
        mutation_rate: Mutation probability
        hidden_size: LSTM hidden size
        num_layers: Number of LSTM layers
        train_dnc: Whether to train DNC operator
        train_mode: 'pretrain' or 'online'
        pretrain_episodes: Number of pretraining episodes
        online_interval: Train DNC every N generations (online mode)
        online_episodes: Episodes per online update
        warmup_gens: Warmup generations before online training
        batch_size: Batch size for RL training
        learning_rate: Learning rate for RL training
        penalty_weight: Penalty weight for invalid individuals
        seed: Random seed
        output_dir: Output directory for results
        gif_fps: GIF frames per second
        snapshot_interval: Save tour snapshot every N generations
    """
    # Set seeds
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_path / f"tsp_{num_cities}cities_{timestamp}"
    run_dir.mkdir(exist_ok=True)

    print("="*80)
    print(f"TSP Genetic Algorithm with DNC/LSTM Crossover")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Cities: {num_cities}")
    print(f"  Population: {population_size}")
    print(f"  Generations: {generations}")
    print(f"  Mutation rate: {mutation_rate}")
    print(f"  DNC hidden size: {hidden_size}")
    print(f"  DNC layers: {num_layers}")
    print(f"  Training mode: {train_mode if train_dnc else 'No training'}")
    print(f"  Penalty weight: {penalty_weight}")
    print(f"  Output directory: {run_dir}")
    print(f"  Seed: {seed}")

    # Initialize TSP problem with correctness checking
    problem = TSPWithCorrectness(num_cities=num_cities, seed=seed, penalty_weight=penalty_weight,
                                 use_soft_penalty=use_soft_penalty)
    print(f"\n[OK] TSP problem initialized (soft_penalty={use_soft_penalty})")
    print(f"  City coordinates saved for visualization")

    # Initialize DNC crossover
    dnc_crossover = DNCrossover(
        gene_size=1,
        hidden_size=hidden_size,
        num_layers=num_layers,
        seed=seed
    )
    print(f"\n[OK] DNC crossover initialized")

    # Initialize swap mutation (traditional random mutation)
    mutation = SwapMutation(chance=mutation_rate, seed=seed)
    print(f"[OK] Swap mutation initialized")

    # Create GA
    ga = GeneticAlgorithm(
        crossover=dnc_crossover,
        mutation=mutation,
        fitness_function=problem.fitness_function,
        seed=seed
    )
    print(f"[OK] GA initialized")

    # Generate initial population
    initial_pop = problem.generate_population(population_size)
    ga.init_population(initial_pop)
    print(f"[OK] Initial population created")

    # Check initial population validity
    initial_valid = sum(1 for ind in initial_pop if problem.is_valid_individual(ind))
    print(f"  Initial validity rate: {initial_valid}/{population_size} ({100*initial_valid/population_size:.1f}%)")

    # Setup DNC training
    trainer = None
    if train_dnc:
        print(f"\n{'='*80}")
        print(f"DNC/LSTM Training Setup")
        print(f"{'='*80}")

        trainer = RLTrainer(
            operator=dnc_crossover,
            fitness_function=problem.fitness_function,
            learning_rate=learning_rate,
            baseline_type="moving_average"
        )

        def pop_generator(batch_size):
            return ga.select_parent_pairs(count=batch_size)

        if train_mode == 'pretrain':
            print(f"\nPre-training DNC for {pretrain_episodes} episodes...")
            print(f"  The LSTM will learn to produce valid individuals through fitness feedback")

            trainer.train(
                population_generator=pop_generator,
                num_episodes=pretrain_episodes,
                batch_size=batch_size,
                temperature=temperature,
                reward_type=reward_type,
                save_interval=max(pretrain_episodes // 5, 1),
                save_path=str(run_dir / "dnc_pretrained.pth"),
                verbose=True
            )

            print(f"\n[OK] Pre-training completed")
            print(f"  Initial avg reward: {trainer.history['rewards'][0]:.4f}")
            print(f"  Final avg reward: {trainer.history['rewards'][-1]:.4f}")
            print(f"  Improvement: {trainer.history['rewards'][-1] - trainer.history['rewards'][0]:.4f}")

        elif train_mode == 'online':
            print(f"\n[OK] Online training configured")
            print(f"  Training every {online_interval} generations")
            print(f"  Episodes per update: {online_episodes}")
            print(f"  Warmup period: {warmup_gens} generations")

    # Tracking for visualization and metrics
    tour_history = []
    distance_history = []
    generation_history = []
    validity_rates = []
    diversity_scores = []

    # Training callback for online learning + metrics tracking
    def training_and_metrics_callback(ga_instance, generation):
        # Track metrics every generation
        current_pop = ga_instance.population
        valid_count = sum(1 for ind in current_pop if problem.is_valid_individual(ind))
        validity_rate = 100.0 * valid_count / len(current_pop)
        validity_rates.append(validity_rate)

        # Calculate diversity
        diversity = calculate_population_diversity(current_pop)
        diversity_scores.append(diversity)

        # Save snapshots for GIF
        if generation % snapshot_interval == 0:
            best_ind = max(current_pop, key=problem.fitness_function)
            if problem.is_valid_individual(best_ind):
                tour_history.append(best_ind.copy())
                distance_history.append(problem.get_tour_distance(best_ind))
                generation_history.append(generation)

        # Online DNC training
        if train_dnc and train_mode == 'online' and trainer is not None:
            if generation >= warmup_gens and (generation - warmup_gens) % online_interval == 0 and generation > 0:
                print(f"\n[Gen {generation}] Training DNC ({online_episodes} episodes)...")
                print(f"  Current validity rate: {validity_rate:.1f}%")

                def pop_gen(batch_size):
                    return ga_instance.select_parent_pairs(count=batch_size)

                trainer.train(
                    population_generator=pop_gen,
                    num_episodes=online_episodes,
                    batch_size=batch_size,
                    temperature=temperature,
                    reward_type=reward_type,
                    save_interval=online_episodes + 1,
                    save_path=str(run_dir / "dnc_online.pth"),
                    verbose=False
                )
                avg_reward = sum(trainer.history['rewards'][-online_episodes:]) / online_episodes
                print(f"  Avg reward: {avg_reward:.4f}")

    # Run GA
    print(f"\n{'='*80}")
    print(f"Running Genetic Algorithm")
    print(f"{'='*80}\n")

    best_individual = ga.start(
        generations=generations,
        selector=selector,
        training_callback=training_and_metrics_callback
    )

    # Final snapshot
    if problem.is_valid_individual(best_individual):
        tour_history.append(best_individual.copy())
        distance_history.append(problem.get_tour_distance(best_individual))
        generation_history.append(generations)

    # Results
    print(f"\n{'='*80}")
    print(f"Results Summary")
    print(f"{'='*80}")

    print(f"\nBest Individual Found:")
    print(f"  Tour: {best_individual}")
    print(f"  Fitness: {problem.fitness_function(best_individual):.4f}")
    print(f"  Valid: {problem.is_valid_individual(best_individual)}")

    if problem.is_valid_individual(best_individual):
        print(f"  Tour distance: {problem.get_tour_distance(best_individual):.2f}")

    print(f"\nFinal Population Statistics:")
    final_pop = ga.population
    final_valid = sum(1 for ind in final_pop if problem.is_valid_individual(ind))
    print(f"  Mean fitness: {ga.history['mean'][-1]:.4f}")
    print(f"  Best fitness: {ga.history['max'][-1]:.4f}")
    print(f"  Validity rate: {final_valid}/{len(final_pop)} ({100*final_valid/len(final_pop):.1f}%)")
    print(f"  Final diversity: {diversity_scores[-1]:.2f}")

    print(f"\nImprovement:")
    print(f"  Initial best fitness: {ga.history['max'][0]:.4f}")
    print(f"  Final best fitness: {ga.history['max'][-1]:.4f}")
    print(f"  Total improvement: {ga.history['max'][-1] - ga.history['max'][0]:.4f}")

    if len(distance_history) > 0:
        print(f"  Initial best distance: {distance_history[0]:.2f}")
        print(f"  Final best distance: {distance_history[-1]:.2f}")
        print(f"  Distance reduction: {distance_history[0] - distance_history[-1]:.2f} ({100*(distance_history[0]-distance_history[-1])/distance_history[0]:.1f}%)")

    # Save metrics
    metrics = {
        'num_cities': num_cities,
        'population_size': population_size,
        'generations': generations,
        'best_fitness': problem.fitness_function(best_individual),
        'best_distance': problem.get_tour_distance(best_individual) if problem.is_valid_individual(best_individual) else None,
        'best_tour': best_individual,
        'is_valid': problem.is_valid_individual(best_individual),
        'initial_validity': initial_valid,
        'final_validity': final_valid,
        'validity_rate': validity_rates,
        'diversity': diversity_scores,
        'ga_history': ga.history,
        'config': {
            'mutation_rate': mutation_rate,
            'hidden_size': hidden_size,
            'num_layers': num_layers,
            'train_mode': train_mode if train_dnc else 'none',
            'penalty_weight': penalty_weight,
            'seed': seed
        }
    }

    metrics_path = run_dir / 'metrics.json'
    with open(metrics_path, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        metrics_serializable = {
            k: (v.tolist() if isinstance(v, np.ndarray) else
                [item.tolist() if isinstance(item, np.ndarray) else item for item in v] if isinstance(v, list) else v)
            for k, v in metrics.items()
        }
        json.dump(metrics_serializable, f, indent=2)
    print(f"\n[OK] Metrics saved to {metrics_path}")

    # Generate GIF
    if len(tour_history) > 1:
        gif_path = run_dir / 'tsp_optimization.gif'
        create_tsp_gif(
            problem.cities,
            tour_history,
            distance_history,
            generation_history,
            str(gif_path),
            fps=gif_fps
        )

    # Generate comprehensive plots
    plot_comprehensive_results(
        ga.history,
        problem,
        best_individual,
        metrics,
        run_dir
    )

    # Save DNC model if trained
    if train_dnc:
        model_path = run_dir / 'dnc_final.pth'
        dnc_crossover.save_model(str(model_path))
        print(f"[OK] DNC model saved to {model_path}")

    print(f"\n{'='*80}")
    print(f"All outputs saved to: {run_dir}")
    print(f"{'='*80}\n")

    return best_individual, metrics


def main():
    parser = argparse.ArgumentParser(
        description="TSP Genetic Algorithm with DNC/LSTM Crossover",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Problem parameters
    parser.add_argument('--num-cities', type=int, default=20,
                       help='Number of cities in TSP')
    parser.add_argument('--penalty-weight', type=float, default=1000.0,
                       help='Penalty weight for invalid individuals')

    # GA parameters
    parser.add_argument('--population-size', type=int, default=100,
                       help='Population size')
    parser.add_argument('--generations', type=int, default=200,
                       help='Number of generations')
    parser.add_argument('--selector', type=int, default=20,
                       help='Number of worst individuals to replace')
    parser.add_argument('--mutation-rate', type=float, default=0.3,
                       help='Mutation probability')

    # DNC parameters
    parser.add_argument('--hidden-size', type=int, default=128,
                       help='LSTM hidden size')
    parser.add_argument('--num-layers', type=int, default=2,
                       help='Number of LSTM layers')

    # Training parameters
    parser.add_argument('--no-train', action='store_true',
                       help='Disable DNC training')
    parser.add_argument('--train-mode', type=str, default='online',
                       choices=['pretrain', 'online'],
                       help='Training mode: pretrain or online')
    parser.add_argument('--pretrain-episodes', type=int, default=100,
                       help='Number of pretraining episodes')
    parser.add_argument('--online-interval', type=int, default=10,
                       help='Train every N generations (online mode)')
    parser.add_argument('--online-episodes', type=int, default=10,
                       help='Episodes per online update')
    parser.add_argument('--warmup-gens', type=int, default=20,
                       help='Warmup generations before online training')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for RL training')
    parser.add_argument('--learning-rate', type=float, default=1e-3,
                       help='Learning rate for RL training')
    parser.add_argument('--use-soft-penalty', action='store_true', default=True,
                       help='Use soft penalty mode for better DNC learning')
    parser.add_argument('--reward-type', type=str, default='absolute',
                       choices=['absolute', 'improvement', 'relative'],
                       help='Reward type for DNC training')
    parser.add_argument('--temperature', type=float, default=1.0,
                       help='Temperature for DNC sampling')

    # Output parameters
    parser.add_argument('--output-dir', type=str, default='tsp_results',
                       help='Output directory')
    parser.add_argument('--gif-fps', type=int, default=2,
                       help='GIF frames per second')
    parser.add_argument('--snapshot-interval', type=int, default=5,
                       help='Save tour snapshot every N generations')

    # Misc
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')

    args = parser.parse_args()

    run_tsp_with_dnc(
        num_cities=args.num_cities,
        population_size=args.population_size,
        generations=args.generations,
        selector=args.selector,
        mutation_rate=args.mutation_rate,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        train_dnc=not args.no_train,
        train_mode=args.train_mode,
        pretrain_episodes=args.pretrain_episodes,
        online_interval=args.online_interval,
        online_episodes=args.online_episodes,
        warmup_gens=args.warmup_gens,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        penalty_weight=args.penalty_weight,
        use_soft_penalty=args.use_soft_penalty,
        reward_type=args.reward_type,
        temperature=args.temperature,
        seed=args.seed,
        output_dir=args.output_dir,
        gif_fps=args.gif_fps,
        snapshot_interval=args.snapshot_interval
    )


if __name__ == "__main__":
    main()
