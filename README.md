# Neural Networks as Genetic Operators

Assignment for HVI - 2025. Paper: [Deep Learning-Based Operators for Evolutionary Algorithms](https://arxiv.org/abs/2407.10477).

## Genetic Algorithm Scripts

**crossovers.py** - contains `BaseCrossover` abstract class and concrete implementations (SinglePoint, MultiPoint, Uniform, Cycle, PMX). Each crossover takes two parents and returns an offspring. Optional `seed` parameter for reproducibility.

**mutations.py** - contains `BaseMutation` abstract class and concrete implementations (Inversion, Swap). Each mutation takes an offspring and returns a mutated version based on the `chance` probability. Optional `seed` parameter for reproducibility.

**ga.py** - main `GeneticAlgorithm` class. Initialize with crossover operator, mutation operator, and fitness function. Call `init_population()` with initial population, then `start()` to run the algorithm. Returns the best individual. Supports population saving/loading and history tracking.
