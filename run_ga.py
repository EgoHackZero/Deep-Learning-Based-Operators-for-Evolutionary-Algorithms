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
from genetic_algorithm.crossovers.gp_crossover import PassthroughCrossover, SubtreeCrossover
from genetic_algorithm.crossovers.variable_length_dnc_crossover import VariableLengthDNCrossover

# mutation operators
from genetic_algorithm.mutations.generic_mutations import (
    InversionMutation, SwapMutation
)
from genetic_algorithm.mutations.bert_mutation import SymbolicRegressionBERTMutation
from genetic_algorithm.bert_trainer import BERTMutationTrainer
from genetic_algorithm.tokenizer import SymbolicRegressionTokenizer

# problems
from problems import get_problem


# operator registries
CROSSOVERS = {
    'single': SinglePointCrossover,
    'multi': MultiPointCrossover,
    'uniform': UniformCrossover,
    'cycle': CycleCrossover,
    'pmx': PMXCrossover,
    'dnc': DNCrossover,
    'dnc-gp': VariableLengthDNCrossover,  # DNC for variable-length GP trees
    'passthrough': PassthroughCrossover,  # For mutation-only GP
    'subtree': SubtreeCrossover  # Simple GP crossover
}

MUTATIONS = {
    'inversion': InversionMutation,
    'swap': SwapMutation,
    'bert': SymbolicRegressionBERTMutation  # BERT mutation for GP trees
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
    elif args.problem in ['symbolic_regression', 'symreg']:
        problem_kwargs = {
            'max_tree_depth': args.max_tree_depth,
            'num_points': args.num_data_points
        }
    else:
        problem_kwargs = {}
    
    problem = get_problem(args.problem, **problem_kwargs)
    print(f"Loaded Problem: {problem.problem_name()}")

    # Create tokenizer if needed (shared between dnc-gp and BERT)
    tokenizer = None
    if args.crossover == 'dnc-gp' or args.mutation == 'bert':
        if args.problem not in ['symbolic_regression', 'symreg']:
            raise ValueError("dnc-gp crossover and BERT mutation only support symbolic_regression problem")

        # Create shared tokenizer
        tokenizer = SymbolicRegressionTokenizer(
            terminal_names=problem.get_terminal_set()
        )
        print(f"\nShared tokenizer created (vocab_size={tokenizer.vocab_size})")

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
    elif args.crossover == 'dnc-gp':
        # Variable-length DNC for GP trees (uses shared tokenizer)
        # Set input_size based on problem (max expected tree size)
        max_tree_size = args.max_tree_depth * 10 if hasattr(args, 'max_tree_depth') else 100
        crossover = VariableLengthDNCrossover(
            tokenizer=tokenizer,
            input_size=max_tree_size,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            seed=args.seed,
            fallback_to_parent=True  # Return parent if offspring is invalid
        )
        print(f"Variable-length DNC initialized (hidden_size={args.hidden_size}, max_tree_size={max_tree_size})")
        if args.load_model:
            crossover.load_model(args.load_model)
            print(f"Loaded model from: {args.load_model}")
    else:
        crossover = CROSSOVERS[args.crossover](seed=args.seed)
        print(f"{args.crossover} crossover initialized")

    # initialize mutation
    if args.mutation == 'bert':
        # BERT mutation uses shared tokenizer

        mutation = SymbolicRegressionBERTMutation(
            tokenizer=tokenizer,
            masking_prob=args.masking_prob,
            temperature=args.temperature,
            epsilon_greedy=args.epsilon_greedy,
            embedding_dim=args.bert_embedding_dim,
            num_heads=args.bert_num_heads,
            num_layers=args.bert_num_layers,
            chance=args.mutation_rate,
            seed=args.seed
        )
        print(f"BERT mutation initialized")
        print(f"  Vocab size: {tokenizer.vocab_size}")
        print(f"  Masking prob: {args.masking_prob}")
        print(f"  Epsilon-greedy: {args.epsilon_greedy}")
        print(f"  Embedding dim: {args.bert_embedding_dim}")

        if args.load_model:
            mutation.load_model(args.load_model)
            print(f"  Loaded model from: {args.load_model}")
    else:
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
    
    # optional: train BERT mutation
    bert_trainer = None
    if args.mutation == 'bert' and args.train_bert:
        bert_trainer = BERTMutationTrainer(
            mutation_operator=mutation,
            fitness_function=problem.fitness_function,
            learning_rate=args.learning_rate,
            baseline_type="moving_average"
        )

        # generator for individuals from current population
        def pop_generator_bert(batch_size):
            return [ind.copy() for ind in random.sample(ga.population, min(batch_size, len(ga.population)))]

        if args.train_mode == 'pretrain':
            # Pre-train once before GA starts
            print(f"\nPre-training BERT Mutation ({args.pretrain_episodes} episodes)")

            bert_trainer.train(
                population_generator=pop_generator_bert,
                num_episodes=args.pretrain_episodes,
                batch_size=args.batch_size,
                temperature=args.temperature,
                reward_type="improvement",
                save_interval=max(args.pretrain_episodes // 5, 1),
                save_path=args.save_model or "models/bert_mutation_pretrained.pth",
                verbose=True
            )

            print(f"Pre-training completed")
            print(f"  Initial reward: {bert_trainer.history['rewards'][0]:.4f}")
            print(f"  Final reward: {bert_trainer.history['rewards'][-1]:.4f}")

    # optional: train DNC
    trainer = None
    if args.crossover in ['dnc', 'dnc-gp'] and args.train_dnc:
        trainer = RLTrainer(
            operator=crossover,
            fitness_function=problem.fitness_function,
            learning_rate=args.learning_rate,
            baseline_type="moving_average"
        )
        
        # generator for parent pairs from current population
        def pop_generator(batch_size):
            pairs = ga.select_parent_pairs(count=batch_size)
            return pairs
        
        if args.train_mode == 'pretrain':
            # Pre-train once before GA starts
            print(f"\nPre-training DNC ({args.pretrain_episodes} episodes)")
            
            trainer.train(
                population_generator=pop_generator,
                num_episodes=args.pretrain_episodes,
                batch_size=args.batch_size,
                temperature=1.0,
                reward_type="improvement",
                save_interval=max(args.pretrain_episodes // 5, 1),
                save_path=args.save_model or "models/dnc_pretrained.pth",
                verbose=True
            )
            
            print(f"Pre-training completed")
            print(f"  Initial reward: {trainer.history['rewards'][0]:.4f}")
            print(f"  Final reward: {trainer.history['rewards'][-1]:.4f}")
        
        elif args.train_mode == 'online':
            # Online training during GA evolution
            print(f"\nDNC will be trained online during GA")
            print(f"  Training every {args.online_interval} generation(s)")
            print(f"  Episodes per update: {args.online_episodes}")
            if args.warmup_gens > 0:
                print(f"  Warmup period: {args.warmup_gens} generations")
    
    # define training callback for online learning
    training_callback = None

    # BERT mutation online training callback
    if args.mutation == 'bert' and args.train_bert and args.train_mode == 'online' and bert_trainer is not None:
        def training_callback(ga_instance, generation):
            # skip warmup period
            if generation < args.warmup_gens:
                return

            # train at specified intervals
            if (generation - args.warmup_gens) % args.online_interval == 0 and generation > 0:
                print(f"\n[Gen {generation}] Training BERT Mutation ({args.online_episodes} episodes)...")

                def pop_gen_bert(batch_size):
                    return [ind.copy() for ind in random.sample(ga_instance.population, min(batch_size, len(ga_instance.population)))]

                bert_trainer.train(
                    population_generator=pop_gen_bert,
                    num_episodes=args.online_episodes,
                    batch_size=args.batch_size,
                    temperature=args.temperature,
                    reward_type="improvement",
                    save_interval=args.online_episodes + 1,  # don't save during online training
                    save_path=args.save_model or "models/bert_mutation_online.pth",
                    verbose=False
                )
                avg_reward = sum(bert_trainer.history['rewards'][-args.online_episodes:]) / args.online_episodes
                print(f"  Avg reward: {avg_reward:.4f}")

    # DNC crossover online training callback
    elif args.crossover in ['dnc', 'dnc-gp'] and args.train_dnc and args.train_mode == 'online' and trainer is not None:
        def training_callback(ga_instance, generation):
            # skip warmup period
            if generation < args.warmup_gens:
                return
            
            # train at specified intervals
            if (generation - args.warmup_gens) % args.online_interval == 0 and generation > 0:
                print(f"\n[Gen {generation}] Training DNC ({args.online_episodes} episodes)...")
                
                def pop_gen(batch_size):
                    return ga_instance.select_parent_pairs(count=batch_size)
                
                trainer.train(
                    population_generator=pop_gen,
                    num_episodes=args.online_episodes,
                    batch_size=args.batch_size,
                    temperature=1.0,
                    reward_type="improvement",
                    save_interval=args.online_episodes + 1,  # don't save during online training
                    save_path=args.save_model or "models/dnc_online.pth",
                    verbose=False
                )
                avg_reward = sum(trainer.history['rewards'][-args.online_episodes:]) / args.online_episodes
                print(f"  Avg reward: {avg_reward:.4f}")
    
    # run GA
    print(f"\nRunning GA ({args.generations} generations)")
    best = ga.start(
        generations=args.generations,
        selector=args.selector,
        training_callback=training_callback
    )
    
    print(f"\nGA completed")
    
    # results
    print(f"\nBest individual: {best}")
    print(f"Best fitness: {problem.fitness_function(best):.4f}")

    # For symbolic regression, also print the expression
    if args.problem in ['symbolic_regression', 'symreg']:
        expr = problem.get_expression_string(best)
        print(f"Expression: {expr}")

    print(f"\nFinal population stats:")
    print(f"Mean fitness: {ga.history['mean'][-1]:.4f}")
    print(f"Max fitness: {ga.history['max'][-1]:.4f}")

    # Print Variable-Length DNC crossover statistics if used
    if args.crossover == 'dnc-gp':
        crossover.print_stats()

    # Save trained model if requested
    if args.mutation == 'bert' and args.save_model and (args.train_bert or args.load_model):
        mutation.save_model(args.save_model)
        print(f"\nBERT mutation model saved to {args.save_model}")
    
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
        choices=['onemax', 'knapsack', 'tsp', 'symbolic_regression', 'symreg'],
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
    
    # Symbolic regression specific parameters
    parser.add_argument(
        '--max-tree-depth',
        type=int,
        default=5,
        help='Maximum tree depth for symbolic regression'
    )
    parser.add_argument(
        '--num-data-points',
        type=int,
        default=50,
        help='Number of data points for symbolic regression'
    )

    # BERT mutation specific parameters
    parser.add_argument(
        '--bert-embedding-dim',
        type=int,
        default=64,
        help='Embedding dimension for BERT mutation'
    )
    parser.add_argument(
        '--bert-num-heads',
        type=int,
        default=4,
        help='Number of attention heads for BERT mutation'
    )
    parser.add_argument(
        '--bert-num-layers',
        type=int,
        default=2,
        help='Number of transformer layers for BERT mutation'
    )
    parser.add_argument(
        '--masking-prob',
        type=float,
        default=0.15,
        help='Probability of masking each node (BERT mutation)'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=1.0,
        help='Temperature for sampling (BERT mutation)'
    )
    parser.add_argument(
        '--epsilon-greedy',
        type=float,
        default=0.1,
        help='Epsilon-greedy exploration rate for BERT mutation (0.0-1.0, default 0.1)'
    )
    parser.add_argument(
        '--train-bert',
        action='store_true',
        help='Enable BERT mutation training (only with --mutation bert)'
    )

    # RL training parameters (only used with --crossover dnc or --mutation bert)
    parser.add_argument(
        '--train-dnc',
        action='store_true',
        help='Enable DNC training (only with --crossover dnc)'
    )
    parser.add_argument(
        '--train-mode',
        type=str,
        choices=['pretrain', 'online'],
        default='pretrain',
        help='pretrain: train once before GA starts | online: train during GA evolution'
    )
    parser.add_argument(
        '--pretrain-episodes',
        type=int,
        default=100,
        help='Number of episodes for pre-training (--train-mode pretrain)'
    )
    parser.add_argument(
        '--online-interval',
        type=int,
        default=10,
        help='Train DNC every N generations (--train-mode online)'
    )
    parser.add_argument(
        '--online-episodes',
        type=int,
        default=5,
        help='Number of episodes per online training update (--train-mode online)'
    )
    parser.add_argument(
        '--warmup-gens',
        type=int,
        default=0,
        help='Wait N generations before starting online training (--train-mode online)'
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
