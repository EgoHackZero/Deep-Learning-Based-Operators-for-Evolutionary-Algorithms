"""
Comprehensive BERT Mutation Training Test

Tests BERT mutation with training to verify it actually learns.
Compares untrained vs trained performance.
"""

import random
import torch
import time
from problems.symbolic_regression import SymbolicRegression
from genetic_algorithm.tokenizer import SymbolicRegressionTokenizer
from genetic_algorithm.mutations.bert_mutation import SymbolicRegressionBERTMutation
from genetic_algorithm.bert_trainer import BERTMutationTrainer
from genetic_algorithm.ga import GeneticAlgorithm


def test_untrained_vs_trained(num_train_episodes=200, population_size=30):
    """
    Compare untrained BERT vs trained BERT mutation performance.

    Args:
        num_train_episodes: Number of training episodes (200 = ~2-3 min on RTX 2050)
        population_size: Population size for testing
    """
    print("=" * 70)
    print("BERT MUTATION TRAINING TEST")
    print("=" * 70)
    print()

    # Setup problem
    problem = SymbolicRegression(
        target_function=lambda x: x**2 + x + 1,  # Target: x^2 + x + 1
        x_range=(-5, 5),
        num_points=30,
        max_tree_depth=4,
        penalty_coefficient=0.001
    )

    print(f"Target function: x^2 + x + 1")
    print(f"Population size: {population_size}")
    print(f"Training episodes: {num_train_episodes}")
    print()

    # Create tokenizer
    tokenizer = SymbolicRegressionTokenizer(
        terminal_names=problem.get_terminal_set()
    )

    # Create UNTRAINED mutation
    print("Creating UNTRAINED BERT mutation...")
    untrained_mutation = SymbolicRegressionBERTMutation(
        tokenizer=tokenizer,
        masking_prob=0.2,
        temperature=1.0,
        epsilon_greedy=0.1,
        embedding_dim=64,
        num_heads=4,
        num_layers=2,
        seed=42
    )

    # Create TRAINED mutation (will be trained)
    print("Creating BERT mutation for training...")
    trained_mutation = SymbolicRegressionBERTMutation(
        tokenizer=tokenizer,
        masking_prob=0.2,
        temperature=1.0,
        epsilon_greedy=0.1,
        embedding_dim=64,
        num_heads=4,
        num_layers=2,
        seed=43
    )

    print()
    print("-" * 70)
    print("PHASE 1: Testing UNTRAINED BERT")
    print("-" * 70)

    # Test untrained BERT
    random.seed(42)
    torch.manual_seed(42)

    untrained_improvements = 0
    untrained_degradations = 0
    untrained_total_improvement = 0.0

    print("Running 30 mutation tests with UNTRAINED model...")
    for i in range(30):
        individual = problem.generate_individual()
        original_fitness = problem.fitness_function(individual)

        mutated = untrained_mutation.perform(individual)
        mutated_fitness = problem.fitness_function(mutated)

        improvement = mutated_fitness - original_fitness
        untrained_total_improvement += improvement

        if improvement > 0.01:
            untrained_improvements += 1
        elif improvement < -0.01:
            untrained_degradations += 1

    untrained_avg_improvement = untrained_total_improvement / 30

    print(f"\nUNTRAINED Results:")
    print(f"  Improvements: {untrained_improvements}/30")
    print(f"  Degradations: {untrained_degradations}/30")
    print(f"  Average improvement: {untrained_avg_improvement:.6f}")
    print(f"  (Close to 0 means random mutations)")

    print()
    print("-" * 70)
    print("PHASE 2: TRAINING BERT MODEL")
    print("-" * 70)
    print()

    # Train BERT mutation
    trainer = BERTMutationTrainer(
        mutation_operator=trained_mutation,
        fitness_function=problem.fitness_function,
        learning_rate=0.001,
        baseline_type="moving_average"
    )

    # Population generator for training
    def population_generator(batch_size):
        individuals = []
        for _ in range(batch_size):
            individuals.append(problem.generate_individual())
        return individuals

    print(f"Training BERT for {num_train_episodes} episodes...")
    print("This may take 2-5 minutes depending on your GPU...")
    print()

    start_time = time.time()

    trained_mutation.train_mode()
    trainer.train(
        population_generator=population_generator,
        num_episodes=num_train_episodes,
        batch_size=16,
        temperature=1.0,
        reward_type="improvement",
        save_interval=max(num_train_episodes // 4, 1),
        save_path="models/bert_test_trained.pth",
        verbose=True
    )
    trained_mutation.eval_mode()

    training_time = time.time() - start_time

    print()
    print(f"Training completed in {training_time:.1f} seconds ({training_time/60:.1f} minutes)")

    # Show training progress
    if len(trainer.history['rewards']) > 0:
        initial_reward = trainer.history['rewards'][0]
        final_reward = trainer.history['rewards'][-1]
        print(f"\nTraining progress:")
        print(f"  Initial avg reward: {initial_reward:.6f}")
        print(f"  Final avg reward: {final_reward:.6f}")
        print(f"  Change: {final_reward - initial_reward:.6f}")

        if final_reward > initial_reward + 0.001:
            print(f"  Status: LEARNING! Model improved during training")
        else:
            print(f"  Status: No clear improvement (may need more episodes)")

    print()
    print("-" * 70)
    print("PHASE 3: Testing TRAINED BERT")
    print("-" * 70)

    # Test trained BERT
    random.seed(42)
    torch.manual_seed(42)

    trained_improvements = 0
    trained_degradations = 0
    trained_total_improvement = 0.0

    print("Running 30 mutation tests with TRAINED model...")
    for i in range(30):
        individual = problem.generate_individual()
        original_fitness = problem.fitness_function(individual)

        mutated = trained_mutation.perform(individual)
        mutated_fitness = problem.fitness_function(mutated)

        improvement = mutated_fitness - original_fitness
        trained_total_improvement += improvement

        if improvement > 0.01:
            trained_improvements += 1
        elif improvement < -0.01:
            trained_degradations += 1

    trained_avg_improvement = trained_total_improvement / 30

    print(f"\nTRAINED Results:")
    print(f"  Improvements: {trained_improvements}/30")
    print(f"  Degradations: {trained_degradations}/30")
    print(f"  Average improvement: {trained_avg_improvement:.6f}")

    print()
    print("=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    print()
    print(f"UNTRAINED BERT:")
    print(f"  Improvements: {untrained_improvements}/30 ({100*untrained_improvements/30:.1f}%)")
    print(f"  Avg improvement: {untrained_avg_improvement:.6f}")
    print()
    print(f"TRAINED BERT:")
    print(f"  Improvements: {trained_improvements}/30 ({100*trained_improvements/30:.1f}%)")
    print(f"  Avg improvement: {trained_avg_improvement:.6f}")
    print()
    print(f"DIFFERENCE:")
    print(f"  Improvement rate: +{trained_improvements - untrained_improvements} mutations")
    print(f"  Avg improvement: {trained_avg_improvement - untrained_avg_improvement:+.6f}")
    print()

    if trained_avg_improvement > untrained_avg_improvement + 0.001:
        print("CONCLUSION: TRAINED model performs BETTER!")
        print("The BERT mutation successfully learned from training.")
    elif trained_avg_improvement < untrained_avg_improvement - 0.001:
        print("CONCLUSION: UNTRAINED model performs better")
        print("May need more training or different hyperparameters.")
    else:
        print("CONCLUSION: No significant difference")
        print("May need more training episodes or the model converged.")

    print()
    return {
        'untrained_improvements': untrained_improvements,
        'trained_improvements': trained_improvements,
        'untrained_avg': untrained_avg_improvement,
        'trained_avg': trained_avg_improvement,
        'training_time': training_time
    }


def quick_ga_test_with_training():
    """
    Quick GA test showing BERT learns during evolution with online training
    """
    print()
    print("=" * 70)
    print("BONUS: QUICK GA WITH ONLINE TRAINING")
    print("=" * 70)
    print()

    random.seed(42)
    torch.manual_seed(42)

    problem = SymbolicRegression(
        target_function=lambda x: x**2,
        x_range=(-5, 5),
        num_points=20,
        max_tree_depth=4
    )

    tokenizer = SymbolicRegressionTokenizer(
        terminal_names=problem.get_terminal_set()
    )

    mutation = SymbolicRegressionBERTMutation(
        tokenizer=tokenizer,
        masking_prob=0.2,
        temperature=1.0,
        epsilon_greedy=0.1,
        embedding_dim=48,
        num_heads=4,
        num_layers=2,
        chance=0.8,
        seed=42
    )

    from genetic_algorithm.crossovers.gp_crossover import PassthroughCrossover
    crossover = PassthroughCrossover(seed=42)

    ga = GeneticAlgorithm(
        crossover=crossover,
        mutation=mutation,
        fitness_function=problem.fitness_function,
        seed=42
    )

    initial_pop = problem.generate_population(40)
    ga.init_population(initial_pop)

    # Setup online trainer
    trainer = BERTMutationTrainer(
        mutation_operator=mutation,
        fitness_function=problem.fitness_function,
        learning_rate=0.001,
        baseline_type="moving_average"
    )

    def pop_generator(batch_size):
        return [ind.copy() for ind in random.sample(ga.population, min(batch_size, len(ga.population)))]

    def training_callback(ga_instance, generation):
        if generation > 0 and generation % 5 == 0:
            print(f"  [Gen {generation}] Training BERT for 10 episodes...")
            mutation.train_mode()
            trainer.train(
                population_generator=pop_generator,
                num_episodes=10,
                batch_size=12,
                temperature=1.0,
                reward_type="improvement",
                save_interval=999,
                save_path="models/bert_online_test.pth",
                verbose=False
            )
            mutation.eval_mode()
            avg_reward = sum(trainer.history['rewards'][-10:]) / 10
            print(f"  [Gen {generation}] Avg reward: {avg_reward:.4f}")

    print("Running GA for 30 generations with online BERT training...")
    print("(Training every 5 generations)")
    print()

    start_time = time.time()
    best = ga.start(
        generations=30,
        selector=10,
        training_callback=training_callback
    )
    ga_time = time.time() - start_time

    print()
    print(f"GA completed in {ga_time:.1f} seconds ({ga_time/60:.1f} minutes)")
    print(f"\nBest fitness achieved: {problem.fitness_function(best):.4f}")
    print(f"Best expression: {problem.get_expression_string(best)}")
    print(f"\nFitness progression:")
    print(f"  Gen 0:  {ga.history['max'][0]:.4f}")
    print(f"  Gen 10: {ga.history['max'][10]:.4f}")
    print(f"  Gen 20: {ga.history['max'][20]:.4f}")
    print(f"  Gen 29: {ga.history['max'][29]:.4f}")
    print(f"  Improvement: {ga.history['max'][29] - ga.history['max'][0]:.4f}")

    print()


if __name__ == '__main__':
    print("\n")
    print("#" * 70)
    print("# COMPREHENSIVE BERT TRAINING TEST")
    print("# Target: RTX 2050 - should complete in 5-10 minutes")
    print("#" * 70)
    print()

    # Main test
    results = test_untrained_vs_trained(num_train_episodes=200, population_size=30)

    # Quick GA test
    quick_ga_test_with_training()

    print()
    print("#" * 70)
    print("# ALL TESTS COMPLETED")
    print("#" * 70)
    print()
