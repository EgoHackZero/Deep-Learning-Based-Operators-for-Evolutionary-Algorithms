import argparse
import random
import torch
import matplotlib.pyplot as plt

from genetic_algorithm.ga import GeneticAlgorithm
from genetic_algorithm.rl_trainer import RLTrainer

# crossover operators
from genetic_algorithm.crossovers.generic_crossovers import (
    SinglePointCrossover, MultiPointCrossover, UniformCrossover
)
from genetic_algorithm.crossovers.permutation_crossovers import (
    CycleCrossover, PMXCrossover
)
from genetic_algorithm.crossovers.dnc_crossover import DNCrossover

# mutation operators
from genetic_algorithm.mutations.generic_mutations import (
    InversionMutation, SwapMutation
)

# problems
from problems import get_problem


# operator registries
CROSSOVERS = {
    'single': SinglePointCrossover,
    'multi': MultiPointCrossover,
    'uniform': UniformCrossover,
    'cycle': CycleCrossover,
    'pmx': PMXCrossover,
    'dnc': DNCrossover
}

MUTATIONS = {
    'inversion': InversionMutation,
    'swap': SwapMutation
}


def plot_results(history: dict, problem_name: str, save_path: str = None):
    """Plot GA evolution results"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    generations = range(len(history['mean']))
    
    # plot mean fitness
    ax1.plot(generations, history['mean'], label='Mean Fitness', alpha=0.7)
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Fitness')
    ax1.set_title(f'{problem_name} - Mean Fitness')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # plot max fitness
    ax2.plot(generations, history['max'], label='Best Fitness', color='green', alpha=0.7)
    ax2.set_xlabel('Generation')
    ax2.set_ylabel('Fitness')
    ax2.set_title(f'{problem_name} - Best Fitness')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"✓ Plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def run_ga(args):
    """Run genetic algorithm with specified configuration"""

    # set seed
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
    
    # prepare problem-specific kwargs
    if args.problem == 'onemax':
        problem_kwargs = {'chromosome_length': args.chromosome_length}
    elif args.problem == 'knapsack':
        problem_kwargs = {'num_items': args.chromosome_length, 'seed': args.seed}
    elif args.problem == 'tsp':
        problem_kwargs = {'num_cities': args.chromosome_length, 'seed': args.seed}
    else:
        problem_kwargs = {}
    
    problem = get_problem(args.problem, **problem_kwargs)
    print(f"Loaded Problem: {problem.problem_name()}")
    
    # initialize crossover
    print(f"\nInitializing crossover: {args.crossover}")
    if args.crossover == 'dnc':
        crossover = DNCrossover(
            gene_size=1,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            model_path=args.load_model,
            seed=args.seed
        )
        print(f"DNC initialized (hidden_size={args.hidden_size})")
        if args.load_model:
            print(f"Loaded model from: {args.load_model}")
    else:
        crossover = CROSSOVERS[args.crossover](seed=args.seed)
        print(f"{args.crossover} crossover initialized")
    
    # initialize mutation
    mutation = MUTATIONS[args.mutation](chance=args.mutation_rate, seed=args.seed)
    print(f"{args.mutation} mutation initialized (rate={args.mutation_rate})")
    
    # create GA
    ga = GeneticAlgorithm(
        crossover=crossover,
        mutation=mutation,
        fitness_function=problem.fitness_function,
        seed=args.seed
    )
    print(f"GA initialized")
    
    # generate initial population
    initial_pop = problem.generate_population(args.population_size)
    ga.init_population(initial_pop)
    print(f"Population initialized (size={args.population_size})")
    
    # optional: train DNC before running GA
    trainer = None
    if args.crossover == 'dnc' and args.train_episodes > 0:
        trainer = RLTrainer(
            operator=crossover,
            fitness_function=problem.fitness_function,
            learning_rate=args.learning_rate,
            baseline_type="moving_average"
        )
        
        # use current population for training
        def pop_generator(batch_size):
            pairs = ga.select_parent_pairs(count=batch_size)
            return pairs
        
        if args.train_strategy == 'once':
            # Strategy 1: One-time pre-training on initial population
            print(f"\nPre-training DNC operator ({args.train_episodes} episodes)")
            print(f"Strategy: One-time training on initial population")
            
            trainer.train(
                population_generator=pop_generator,
                num_episodes=args.train_episodes,
                batch_size=args.batch_size,
                temperature=1.0,
                reward_type="improvement",
                save_interval=max(args.train_episodes // 5, 1),
                save_path=args.save_model or "models/dnc_trained.pth",
                verbose=True
            )
            
            print(f"DNC pre-training completed")
            print(f"Initial reward: {trainer.history['rewards'][0]:.4f}")
            print(f"Final reward: {trainer.history['rewards'][-1]:.4f}")
        
        elif args.train_strategy in ['periodic', 'continuous']:
            # Strategy 2/3: Online training during GA evolution
            interval = 1 if args.train_strategy == 'continuous' else args.train_interval
            print(f"\nDNC will be trained online during GA")
            print(f"Strategy: {args.train_strategy}")
            print(f"Training interval: every {interval} generation(s)")
            print(f"Episodes per update: {args.train_epochs_per_update}")
            if args.warmup_generations > 0:
                print(f"Warmup period: {args.warmup_generations} generations")
    
    # define training callback for online learning
    training_callback = None
    if args.crossover == 'dnc' and args.train_strategy in ['periodic', 'continuous'] and trainer is not None:
        interval = 1 if args.train_strategy == 'continuous' else args.train_interval
        
        def training_callback(ga_instance, generation):
            # skip warmup period
            if generation < args.warmup_generations:
                return
            
            # train at specified intervals
            if (generation - args.warmup_generations) % interval == 0 and generation > 0:
                print(f"\n[Gen {generation}] Training DNC for {args.train_epochs_per_update} epochs...")
                
                def pop_gen(batch_size):
                    return ga_instance.select_parent_pairs(count=batch_size)
                
                trainer.train(
                    population_generator=pop_gen,
                    num_episodes=args.train_epochs_per_update,
                    batch_size=args.batch_size,
                    temperature=1.0,
                    reward_type="improvement",
                    save_interval=args.train_epochs_per_update + 1,  # don't save during online training
                    save_path=args.save_model or "models/dnc_online.pth",
                    verbose=False
                )
                avg_reward = sum(trainer.history['rewards'][-args.train_epochs_per_update:]) / args.train_epochs_per_update
                print(f"Avg reward: {avg_reward:.4f}")
    
    # run GA
    step_num = '7' if (args.crossover == 'dnc' and args.train_episodes > 0) else '6'
    print(f"\n[{step_num}] Running GA ({args.generations} generations)")
    best = ga.start(
        generations=args.generations,
        selector=args.selector,
        training_callback=training_callback
    )
    
    print(f"\nGA completed")
    
    # results
    print(f"\nBest individual: {best}")
    print(f"Best fitness: {problem.fitness_function(best):.4f}")
    print(f"\nFinal population stats:")
    print(f"Mean fitness: {ga.history['mean'][-1]:.4f}")
    print(f"Max fitness: {ga.history['max'][-1]:.4f}")
    
    # plot results
    if args.plot:
        print(f"\nGenerating plots")
        plot_results(
            ga.history,
            problem.problem_name(),
            save_path="ga_results.png"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Run Genetic Algorithm with various operators and problems",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # problem configuration
    parser.add_argument(
        '--problem',
        type=str,
        choices=['onemax', 'knapsack', 'tsp'],
        default='onemax',
        help='Problem to solve'
    )
    parser.add_argument(
        '--chromosome-length',
        type=int,
        default=50,
        help='Length of chromosome (problem-dependent)'
    )
    
    # GA parameters
    parser.add_argument(
        '--population-size',
        type=int,
        default=100,
        help='Population size'
    )
    parser.add_argument(
        '--generations',
        type=int,
        default=100,
        help='Number of generations'
    )
    parser.add_argument(
        '--selector',
        type=int,
        default=10,
        help='Number of worst individuals to replace'
    )
    
    # operator selection
    parser.add_argument(
        '--crossover',
        type=str,
        choices=list(CROSSOVERS.keys()),
        default='single',
        help='Crossover operator'
    )
    parser.add_argument(
        '--mutation',
        type=str,
        choices=list(MUTATIONS.keys()),
        default='swap',
        help='Mutation operator'
    )
    parser.add_argument(
        '--mutation-rate',
        type=float,
        default=0.1,
        help='Mutation probability'
    )
    
    # DNC-specific parameters
    parser.add_argument(
        '--hidden-size',
        type=int,
        default=64,
        help='Hidden size for DNC (if using DNC crossover)'
    )
    parser.add_argument(
        '--num-layers',
        type=int,
        default=1,
        help='Number of LSTM layers for DNC'
    )
    parser.add_argument(
        '--load-model',
        type=str,
        default=None,
        help='Path to pretrained DNC model'
    )
    
    # RL training parameters
    parser.add_argument(
        '--train-episodes',
        type=int,
        default=0,
        help='Number of RL training episodes (0 = no training)'
    )
    parser.add_argument(
        '--train-strategy',
        type=str,
        choices=['once', 'periodic', 'continuous'],
        default='once',
        help='Training strategy: once (pre-train), periodic (every N gens), continuous (every gen)'
    )
    parser.add_argument(
        '--train-interval',
        type=int,
        default=10,
        help='Training interval for periodic strategy (train every N generations)'
    )
    parser.add_argument(
        '--train-epochs-per-update',
        type=int,
        default=5,
        help='Number of training epochs per update for periodic/continuous strategies'
    )
    parser.add_argument(
        '--warmup-generations',
        type=int,
        default=0,
        help='Number of generations to run before starting DNC training'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for RL training'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=1e-3,
        help='Learning rate for RL training'
    )
    parser.add_argument(
        '--save-model',
        type=str,
        default=None,
        help='Path to save trained DNC model'
    )
    
    # misc
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )
    parser.add_argument(
        '--plot',
        action='store_true',
        help='Generate and save plots'
    )
    
    args = parser.parse_args()
    
    # run GA
    run_ga(args)


if __name__ == "__main__":
    main()
