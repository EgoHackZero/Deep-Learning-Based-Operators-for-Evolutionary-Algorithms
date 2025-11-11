# Neural Networks as Genetic Operators

Assignment for HVI - 2025. Paper: [Deep Learning-Based Operators for Evolutionary Algorithms](https://arxiv.org/abs/2407.10477).

## Genetic Algorithm Scripts

**genetic_algorithm/crossovers/** - contains `BaseCrossover` abstract class and concrete implementations (SinglePoint, MultiPoint, Uniform, Cycle, PMX, DNC). Traditional crossovers are deterministic, DNC is LSTM-based and trainable via RL.

**genetic_algorithm/mutations/** - contains `BaseMutation` abstract class and concrete implementations (Inversion, Swap). Each mutation takes an offspring and returns a mutated version based on the `chance` probability.

**genetic_algorithm/ga.py** - main `GeneticAlgorithm` class. Initialize with crossover operator, mutation operator, and fitness function. Call `init_population()` with initial population, then `start()` to run the algorithm. Returns the best individual.

**genetic_algorithm/rl_trainer.py** - `RLTrainer` class for training DL-based operators using REINFORCE (policy gradient). Trains operators to maximize offspring fitness improvement over parents.

**problems/** - problem definitions. Contains `BaseProblem` abstract class and example implementations (OneMax, Knapsack, TSP). Define custom problems by inheriting from `BaseProblem` and implementing `fitness_function()` and `generate_individual()`.

**run_ga.py** - main execution script with CLI. Run GA with any combination of operators and problems.

## Quick Start

```bash
# Basic usage with traditional operators
python run_ga.py --problem onemax --crossover single --mutation swap --generations 100

# Using DNC crossover (untrained)
python run_ga.py --problem tsp --crossover dnc --generations 50

# Pre-train DNC then run GA
python run_ga.py --crossover dnc --train-dnc --train-mode pretrain --pretrain-episodes 200 --generations 100

# Online training (DNC adapts during GA evolution)
python run_ga.py --crossover dnc --train-dnc --train-mode online --online-interval 10 --warmup-gens 20

# Full parameter list
python run_ga.py --help
```

## Custom Problems

Create a new problem class inheriting from `BaseProblem`:

```python
from problems.base_problem import BaseProblem

class MyProblem(BaseProblem):
    def fitness_function(self, individual: list) -> float:
        # return fitness score (higher is better)
        return sum(individual)
    
    def generate_individual(self) -> list:
        # return random individual
        return [random.randint(0, 1) for _ in range(50)]
    
    def problem_name(self) -> str:
        return "My Problem"
```

Register in `problems/__init__.py`:
```python
from problems.my_problem import MyProblem

PROBLEMS = {
    'myproblem': MyProblem,
    # ... existing problems
}
```
