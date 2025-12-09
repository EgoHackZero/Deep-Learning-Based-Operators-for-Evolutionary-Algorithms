# Neural Networks as Genetic Operators

Research project implementing Neural Networks as Genetic Operators for HVI - 2025. Based on the paper [Deep Learning-Based Operators for Evolutionary Algorithms](https://arxiv.org/abs/2407.10477).

This codebase implements a flexible genetic algorithm framework with both traditional and deep learning-based operators:
- **Traditional operators**: Single/Multi/Uniform crossover, Cycle/PMX crossover, Swap/Inversion mutation
- **DNC (Deep Neural Crossover)**: LSTM-based trainable crossover operator for fixed-length individuals
- **BERT Mutation**: Transformer-based trainable mutation operator for variable-length GP trees
- **RL Training**: REINFORCE-based training infrastructure for neural operators
- **Problem abstraction**: Supports both fixed-length (OneMax, TSP, Knapsack) and variable-length GP trees (Symbolic Regression)

## Quick Start

```bash
# Basic GA with traditional operators
python run_ga.py --problem onemax --crossover single --mutation swap --generations 100

# Using DNC crossover (untrained)
python run_ga.py --problem tsp --crossover dnc --hidden-size 64 --generations 50

# Pre-train DNC then run GA
python run_ga.py --crossover dnc --train-dnc --train-mode pretrain --pretrain-episodes 200 --generations 100

# Online training (DNC adapts during evolution)
python run_ga.py --crossover dnc --train-dnc --train-mode online --online-interval 10 --warmup-gens 20

# BERT mutation for symbolic regression (GP trees)
python run_ga.py --problem symreg --mutation bert --crossover passthrough --generations 50

# Use both DNC-GP and BERT together (shared tokenizer)
python run_ga.py --problem symreg --crossover dnc-gp --mutation bert \
  --train-dnc --train-bert --train-mode online --online-interval 10 \
  --warmup-gens 20 --generations 100

# Pre-train BERT mutation then run GA
python run_ga.py --problem symreg --mutation bert --crossover passthrough \
  --train-bert --train-mode pretrain --pretrain-episodes 100 \
  --save-model models/bert_mutation.pth --generations 50

# Load pre-trained BERT mutation
python run_ga.py --problem symreg --mutation bert --crossover passthrough \
  --load-model models/bert_mutation.pth --generations 50

# Full parameter list
python run_ga.py --help
```

## Project Structure

```
genetic_algorithm/
├── ga.py                  - Main GeneticAlgorithm class
├── rl_trainer.py          - RLTrainer for DL-based crossover operators
├── bert_trainer.py        - BERTMutationTrainer for DL-based mutation operators
├── gp_tree.py             - GP tree representation (prefix notation, metadata)
├── tokenizer.py           - Tokenization for BERT (GP trees → tokens)
├── bert_model.py          - Transformer model for masked node prediction
├── crossovers/
│   ├── base_crossover.py        - BaseCrossover abstract class
│   ├── generic_crossovers.py    - SinglePoint, MultiPoint, Uniform (fixed-length)
│   ├── permutation_crossovers.py - Cycle, PMX (for TSP-like problems)
│   ├── gp_crossover.py          - Passthrough, Subtree (variable-length GP)
│   └── dnc_crossover.py         - DNCrossover (LSTM-based, trainable)
└── mutations/
    ├── base_mutation.py         - BaseMutation abstract class
    ├── generic_mutations.py     - Inversion, Swap (fixed-length)
    └── bert_mutation.py         - BERTMutation (Transformer-based, trainable, GP)

problems/
├── base_problem.py        - BaseProblem abstract class
├── onemax.py              - Binary optimization
├── knapsack.py            - 0-1 knapsack problem
├── tsp.py                 - Traveling salesman problem
└── symbolic_regression.py - Symbolic regression (GP tree problem)

run_ga.py                  - Main CLI script with full configuration
```

## Core Components

### GeneticAlgorithm (ga.py)
Main evolution loop with tournament selection and generational replacement.

**Key methods:**
- `init_population(initial_pop)` - Initialize population
- `start(generations, selector, training_callback=None)` - Run evolution
- `select_parent_pairs(count, tournament_size=2)` - Select parents for training
- `history` property - Contains fitness tracking `{"mean": [...], "max": [...]}`

**Selection & Replacement:**
- Tournament selection (default tournament_size=2)
- Generational replacement removes `selector` worst individuals each generation

### Deep Learning-Based Operators

#### DNCrossover (dnc_crossover.py)
LSTM encoder-decoder architecture for fixed-length individuals.

**Architecture:**
- Encoder processes both parents → hidden representations
- Decoder outputs selection probabilities (parent1 vs parent2 for each gene)

**Methods:**
- `perform(parent1, parent2, temperature=1.0)` - Generate offspring
- `train_mode()` / `eval_mode()` - Switch between training/inference
- `load_model(path)` / `save_model(path)` - Model persistence

**Training:**
- Trained with RLTrainer using REINFORCE (policy gradient)
- Reward = offspring_fitness - avg(parent_fitness)
- Supports pre-training or online training during evolution

#### BERTMutation (bert_mutation.py)
Transformer-based architecture for variable-length GP trees.

**Architecture:**
- Transformer encoder with multi-head self-attention
- Masks random nodes, predicts replacements using BERT model
- Sequential mask replacement in DFS order (each replacement influences next)
- Type-safe constraints: only valid replacements (terminal/function/arity)

**Methods:**
- `perform(individual)` - Apply mutation (inherited from BaseMutation)
- `train_mode()` / `eval_mode()` - Switch between training/inference
- `load_model(path)` / `save_model(path)` - Model persistence

**Training:**
- Trained with BERTMutationTrainer using REINFORCE
- Reward = fitness(mutated) - fitness(original)
- Sequential replacement log probabilities for policy gradient
- Type constraints ensure valid GP trees

### Genetic Programming (GP)

#### GP Tree Representation (gp_tree.py)
Trees stored in prefix notation: `['*', '+', 'x', 'y', 2.0]` = (x + y) * 2

**GPTree class:**
- Validates structure and tracks metadata (node type, arity)
- DFS traversal determines mask replacement order for BERT mutation
- Supports variable-length individuals

#### Tokenization (tokenizer.py)
Converts GP trees to token sequences for neural operators.

**BaseTokenizer:**
- Abstract class for problem-specific vocabularies
- Maintains metadata for type-safe replacements
- Special tokens: [PAD], [MASK], [UNK]
- **IMPORTANT**: Shared between BERT mutation and DNC-GP crossover for consistent vocabularies

**SymbolicRegressionTokenizer:**
- Handles functions (+, -, *, /), terminals (x, y), constants
- Problem-specific vocabulary for symbolic regression

**Unified Tokenization Architecture:**
When using `dnc-gp` crossover or `bert` mutation, both operators share the same tokenizer instance. This ensures:
- Consistent token IDs across operators
- No vocabulary conflicts
- More efficient memory usage
- Better interoperability when using both operators together

## Operator Interfaces

### BaseCrossover
All crossovers implement `perform(parent1, parent2, **kwargs) -> offspring`

**Traditional crossovers:**
- Validate parents have equal length
- SinglePoint, MultiPoint, Uniform, Cycle, PMX

**GP crossovers:**
- Support variable-length trees
- Passthrough (returns parent1 unchanged)
- Subtree (swaps random subtrees)

### BaseMutation
All mutations implement `_mutate(offspring, **kwargs) -> mutated_offspring`

**Wrapper method:**
- `perform()` applies mutation based on `chance` probability

**Traditional mutations:**
- Fixed-length lists: Inversion, Swap

**GP mutations:**
- Variable-length trees: BERT mutation

## Problem Interface

### Creating Custom Problems

#### Fixed-Length Problems
Define custom problems by inheriting from `BaseProblem`:

```python
from problems.base_problem import BaseProblem
import random

class MyProblem(BaseProblem):
    def fitness_function(self, individual: list) -> float:
        # Return fitness score (higher is better)
        return sum(individual)

    def generate_individual(self) -> list:
        # Return random individual
        return [random.randint(0, 1) for _ in range(50)]

    def problem_name(self) -> str:
        return "My Problem"
```

#### GP Problems (Variable-Length Trees)
For GP problems, additionally implement:

```python
from problems.base_problem import BaseProblem
import random

class MyGPProblem(BaseProblem):
    def fitness_function(self, individual: list) -> float:
        # Evaluate GP tree (prefix notation)
        # Return fitness score (higher is better)
        pass

    def generate_individual(self) -> list:
        # Generate random GP tree in prefix notation
        pass

    def problem_name(self) -> str:
        return "My GP Problem"

    def get_function_arities(self) -> dict:
        # Return function names to arities
        return {'+': 2, '-': 2, '*': 2, '/': 2}

    def get_terminal_set(self) -> list:
        # Return terminal symbols
        return ['x', 'y']
```

#### Register Problem
Add to `problems/__init__.py`:

```python
from problems.my_problem import MyProblem

PROBLEMS = {
    'myproblem': MyProblem,
    # ... existing problems
}
```

## Training Modes

### Pre-training
Train operators once before GA starts using initial population.

```bash
python run_ga.py --crossover dnc --train-dnc --train-mode pretrain --pretrain-episodes 200
```

### Online Training
Periodically train operators during GA evolution (use `warmup_gens` to stabilize population first).

```bash
python run_ga.py --crossover dnc --train-dnc --train-mode online --online-interval 10 --warmup-gens 20
```

### Save/Load Models

```bash
# Save trained model
python run_ga.py --crossover dnc --train-dnc --save-model models/my_dnc.pth

# Load pre-trained model
python run_ga.py --crossover dnc --load-model models/my_dnc.pth
```

## Important Implementation Details

**Fitness Convention:**
- Higher values = better fitness (used in all `max()` calls)
- All built-in problems follow this convention

**Reproducibility:**
- Set `--seed` for deterministic results
- Seeds apply to: random, torch, and all operators

**When to Use BERT Mutation:**
- GP problems with variable-length tree representations
- Symbolic regression, program synthesis
- Use `--crossover passthrough` for pure mutation-based evolution
- Or `--crossover subtree` for simple GP crossover
- Traditional fixed-length crossovers will fail on variable-length trees

**Key Differences from Standard BERT:**
- No ground-truth labels → uses REINFORCE for training
- Sequential replacement in DFS order (not simultaneous)
- Type constraints (terminals vs functions, arity matching)
- Reward = fitness improvement (not cross-entropy loss)

## Code Origins

Base GA implementation adapted from [github.com/Mruzik1/Genetic-Algorithms](https://github.com/Mruzik1/Genetic-Algorithms/tree/main). DNC operator, BERT mutation operator, and RL training infrastructure implemented for this assignment.
