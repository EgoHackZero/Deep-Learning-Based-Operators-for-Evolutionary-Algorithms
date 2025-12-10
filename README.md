# Neural Networks as Genetic Operators

Research project implementing Neural Networks as Genetic Operators for HVI - 2025. Based on the paper [Deep Learning-Based Operators for Evolutionary Algorithms](https://arxiv.org/abs/2407.10477).

This codebase implements a flexible genetic algorithm framework with both traditional and deep learning-based operators:
- **Traditional operators**: Single/Multi/Uniform crossover, Cycle/PMX crossover, Swap/Inversion mutation
- **DNC (Deep Neural Crossover)**: LSTM-based trainable crossover operator for fixed-length individuals
- **Variable-Length DNC**: Adapted DNC for variable-length GP trees with padding and validation
- **BERT Mutation**: Transformer-based trainable mutation operator for variable-length GP trees
- **RL Training**: REINFORCE-based training infrastructure for neural operators
- **Problem abstraction**: Supports both fixed-length (OneMax, TSP, Knapsack) and variable-length GP trees (Symbolic Regression)

## Quick Start

```bash
# Basic GA with traditional operators
python run_ga.py --problem onemax --crossover single --mutation swap --generations 100

# Using DNC crossover (untrained) on TSP
python run_ga.py --problem tsp --crossover dnc --hidden-size 64 --generations 50

# Pre-train DNC then run GA
python run_ga.py --crossover dnc --train-dnc --train-mode pretrain --pretrain-episodes 200 --generations 100

# Online training (DNC adapts during evolution)
python run_ga.py --crossover dnc --train-dnc --train-mode online --online-interval 10 --warmup-gens 20

# BERT mutation for symbolic regression (GP trees)
python run_ga.py --problem symreg --mutation bert --crossover passthrough --generations 50

# Variable-Length DNC for symbolic regression (GP trees)
python run_ga.py --problem symreg --crossover dnc-gp --mutation swap --hidden-size 128 --generations 100

# Use both Variable-Length DNC and BERT together (shared tokenizer)
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
nn_genetic_operators/
├── genetic_algorithm/
│   ├── ga.py                           - Main GeneticAlgorithm class
│   ├── rl_trainer.py                   - RLTrainer for DNC crossover operators
│   ├── bert_trainer.py                 - BERTMutationTrainer for BERT mutation
│   ├── gp_tree.py                      - GP tree representation (prefix notation, validation)
│   ├── tokenizer.py                    - Tokenization for BERT/DNC-GP (shared vocabulary)
│   ├── bert_model.py                   - Transformer model for masked node prediction
│   ├── crossovers/
│   │   ├── base_crossover.py           - BaseCrossover abstract class
│   │   ├── generic_crossovers.py       - Single/Multi/Uniform (fixed-length)
│   │   ├── permutation_crossovers.py   - Cycle/PMX (for TSP-like problems)
│   │   ├── dnc_crossover.py            - DNCrossover (LSTM-based, trainable, fixed-length)
│   │   ├── variable_length_dnc_crossover.py - VariableLengthDNCrossover (DNC for GP trees)
│   │   ├── gp_crossover.py             - Passthrough/Subtree (variable-length GP)
│   │   └── __init__.py
│   └── mutations/
│       ├── base_mutation.py            - BaseMutation abstract class
│       ├── generic_mutations.py        - Inversion/Swap (fixed-length)
│       ├── bert_mutation.py            - BERTMutation & SymbolicRegressionBERTMutation
│       └── __init__.py
│
├── problems/
│   ├── base_problem.py                 - BaseProblem abstract class
│   ├── onemax.py                       - Binary optimization
│   ├── knapsack.py                     - 0-1 knapsack problem
│   ├── tsp.py                          - Traveling salesman problem
│   ├── symbolic_regression.py          - Symbolic regression (GP tree problem)
│   └── __init__.py
│
├── models/                             - Directory for saved models
├── run_ga.py                           - Main CLI script with full configuration
├── run_tsp_with_dnc.py                 - Specialized TSP+DNC example
├── test_bert_training.py               - BERT mutation training tests
└── test_trained_bert_ga.py             - BERT GA integration tests
```

## Core Components

### GeneticAlgorithm (ga.py)
Main evolution loop with tournament selection and generational replacement.

**Key methods:**
- `init_population(initial_pop)` - Initialize population
- `start(generations, selector, training_callback=None)` - Run evolution with optional training callback
- `select_parent_pairs(count, tournament_size=2)` - Select parents for training
- `history` property - Contains fitness tracking `{"mean": [...], "max": [...]}`

**Selection & Replacement:**
- Tournament selection (default tournament_size=2)
- Generational replacement removes `selector` worst individuals each generation

### Deep Learning-Based Operators

#### DNCrossover (dnc_crossover.py)
LSTM encoder-decoder architecture for **fixed-length** individuals.

**Architecture:**
- Encoder LSTM processes both parents → hidden representations
- Decoder LSTM outputs selection probabilities (parent1 vs parent2 for each gene)
- Supports multi-layer LSTMs

**Methods:**
- `perform(parent1, parent2, temperature=1.0)` - Generate offspring
- `train_mode()` / `eval_mode()` - Switch between training/inference
- `load_model(path)` / `save_model(path)` - Model persistence

**Training:**
- Trained with RLTrainer using REINFORCE (policy gradient)
- Reward = offspring_fitness - avg(parent_fitness)
- Supports pre-training or online training during evolution

#### VariableLengthDNCrossover (variable_length_dnc_crossover.py)
Adapted DNC for **variable-length GP trees**.

**Key Features:**
- Uses shared tokenizer with BERT mutation for consistent vocabularies
- Pads shorter parent to match longer parent's length
- Validates offspring as valid GP trees using GPTree class
- Falls back to parent1 if offspring is invalid (configurable)
- Tracks success statistics (valid offspring ratio)

**Architecture:**
- Wraps DNCrossover with tokenization and padding logic
- Ensures type-safe operations on GP trees
- Supports same training modes as DNC

**Methods:**
- `perform(parent1, parent2, temperature=1.0)` - Generate offspring with validation
- `print_stats()` - Display offspring validity statistics
- `train_mode()` / `eval_mode()` - Switch between training/inference
- `load_model(path)` / `save_model(path)` - Model persistence

#### BERTMutation (bert_mutation.py)
Transformer-based architecture for variable-length GP trees.

**Architecture:**
- Transformer encoder with multi-head self-attention
- Masks random nodes with configurable probability
- Predicts replacements using BERT model (MLM head)
- Sequential mask replacement in DFS order (each replacement influences next)
- Type-safe constraints: only valid replacements (terminal/function/arity)

**Concrete Implementation:**
- `SymbolicRegressionBERTMutation` - Specialized for symbolic regression trees
- Integrates with SymbolicRegressionTokenizer

**Methods:**
- `perform(individual)` - Apply mutation (inherited from BaseMutation)
- `train_mode()` / `eval_mode()` - Switch between training/inference
- `load_model(path)` / `save_model(path)` - Model persistence

**Training:**
- Trained with BERTMutationTrainer using REINFORCE
- Reward = fitness(mutated) - fitness(original)
- Sequential replacement log probabilities for policy gradient
- Type constraints ensure valid GP trees

**Hyperparameters:**
- `masking_prob` (default: 0.15) - Probability of masking each node
- `temperature` (default: 1.0) - Sampling temperature
- `epsilon_greedy` (default: 0.1) - Exploration rate

### Genetic Programming (GP)

#### GP Tree Representation (gp_tree.py)
Trees stored in prefix notation: `['*', '+', 'x', 'y', 2.0]` = (x + y) * 2

**GPTree class:**
- Validates structure and tracks metadata (node type, arity)
- DFS traversal determines mask replacement order for BERT mutation
- Supports variable-length individuals
- Type-aware node validation

#### Tokenization (tokenizer.py)
Converts GP trees to token sequences for neural operators.

**BaseTokenizer:**
- Abstract class for problem-specific vocabularies
- Maintains metadata for type-safe replacements
- Special tokens: [PAD], [MASK], [UNK]
- **IMPORTANT**: Shared between BERT mutation and Variable-Length DNC for consistent vocabularies

**SymbolicRegressionTokenizer:**
- Handles functions (+, -, *, /), terminals (x, y), constants
- Problem-specific vocabulary for symbolic regression
- Provides node type information (function/terminal/constant)

**Unified Tokenization Architecture:**
When using `dnc-gp` crossover or `bert` mutation, both operators share the same tokenizer instance. This ensures:
- Consistent token IDs across operators
- No vocabulary conflicts
- More efficient memory usage
- Better interoperability when using both operators together

## Operator Interfaces

### BaseCrossover
All crossovers implement `perform(parent1, parent2, **kwargs) -> offspring`

**Traditional crossovers (fixed-length):**
- Validate parents have equal length
- SinglePoint, MultiPoint, Uniform, Cycle, PMX

**GP crossovers (variable-length):**
- Support variable-length trees
- Passthrough (returns parent1 unchanged)
- Subtree (swaps random subtrees)
- Variable-Length DNC (LSTM-based, trainable)

### BaseMutation
All mutations implement `_mutate(offspring, **kwargs) -> mutated_offspring`

**Wrapper method:**
- `perform()` applies mutation based on `chance` probability

**Traditional mutations:**
- Fixed-length lists: Inversion, Swap

**GP mutations:**
- Variable-length trees: BERT mutation (SymbolicRegressionBERTMutation)

## Problem Interface

### Built-in Problems

#### Fixed-Length Problems
1. **OneMax** (`onemax`)
   - Binary string optimization: maximize count of 1s
   - Parameters: `--chromosome-length` (default: 50)

2. **Knapsack** (`knapsack`)
   - 0-1 knapsack optimization with capacity constraint
   - Parameters: `--chromosome-length` (num_items, default: 50)

3. **TSP** (`tsp`)
   - Traveling salesman problem: minimize tour distance
   - Parameters: `--chromosome-length` (num_cities, default: 50)

#### Variable-Length Problem (GP)
1. **Symbolic Regression** (`symbolic_regression` or `symreg`)
   - Evolves mathematical expressions to fit target data
   - Target function: x² + x + 1 (configurable in code)
   - Parameters:
     - `--max-tree-depth` (default: 5)
     - `--num-data-points` (default: 50)
   - Function set: +, -, *, /
   - Terminal set: ['x']

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
# Pre-train DNC
python run_ga.py --crossover dnc --train-dnc --train-mode pretrain --pretrain-episodes 200

# Pre-train BERT mutation
python run_ga.py --problem symreg --mutation bert --crossover passthrough \
  --train-bert --train-mode pretrain --pretrain-episodes 100
```

### Online Training
Periodically train operators during GA evolution (use `warmup-gens` to stabilize population first).

```bash
# DNC online training
python run_ga.py --crossover dnc --train-dnc --train-mode online \
  --online-interval 10 --online-episodes 5 --warmup-gens 20

# BERT mutation online training
python run_ga.py --problem symreg --mutation bert --train-bert --train-mode online \
  --online-interval 10 --online-episodes 5 --warmup-gens 20

# Both operators trained online
python run_ga.py --problem symreg --crossover dnc-gp --mutation bert \
  --train-dnc --train-bert --train-mode online --online-interval 10 --warmup-gens 20
```

### Save/Load Models

```bash
# Save trained DNC model
python run_ga.py --crossover dnc --train-dnc --save-model models/my_dnc.pth

# Load pre-trained DNC model
python run_ga.py --crossover dnc --load-model models/my_dnc.pth

# Save trained BERT model
python run_ga.py --problem symreg --mutation bert --train-bert \
  --save-model models/my_bert.pth

# Load pre-trained BERT model
python run_ga.py --problem symreg --mutation bert \
  --load-model models/my_bert.pth
```

## CLI Arguments Reference

### Problem Configuration
- `--problem` - Problem to solve: `onemax`, `knapsack`, `tsp`, `symbolic_regression`, `symreg` (default: `onemax`)
- `--chromosome-length` - Length of chromosome for fixed-length problems (default: 50)
- `--max-tree-depth` - Maximum tree depth for symbolic regression (default: 5)
- `--num-data-points` - Number of data points for symbolic regression (default: 50)

### GA Parameters
- `--population-size` - Population size (default: 100)
- `--generations` - Number of generations (default: 100)
- `--selector` - Number of worst individuals to replace each generation (default: 10)

### Operator Selection
- `--crossover` - Crossover operator: `single`, `multi`, `uniform`, `cycle`, `pmx`, `dnc`, `dnc-gp`, `passthrough`, `subtree` (default: `single`)
- `--mutation` - Mutation operator: `swap`, `inversion`, `bert` (default: `swap`)
- `--mutation-rate` - Mutation probability (default: 0.1)

### DNC Parameters
- `--hidden-size` - Hidden size for DNC LSTM (default: 64)
- `--num-layers` - Number of LSTM layers for DNC (default: 1)

### BERT Mutation Parameters
- `--bert-embedding-dim` - Embedding dimension (default: 64)
- `--bert-num-heads` - Number of attention heads (default: 4)
- `--bert-num-layers` - Number of transformer layers (default: 2)
- `--masking-prob` - Probability of masking each node (default: 0.15)
- `--temperature` - Sampling temperature (default: 1.0)
- `--epsilon-greedy` - Exploration rate (default: 0.1)

### Training Parameters
- `--train-dnc` - Enable DNC training (use with `--crossover dnc` or `dnc-gp`)
- `--train-bert` - Enable BERT mutation training (use with `--mutation bert`)
- `--train-mode` - Training mode: `pretrain` or `online` (default: `pretrain`)
- `--pretrain-episodes` - Number of pre-training episodes (default: 100)
- `--online-interval` - Train every N generations in online mode (default: 10)
- `--online-episodes` - Episodes per online training update (default: 5)
- `--warmup-gens` - Wait N generations before starting online training (default: 0)
- `--batch-size` - Batch size for RL training (default: 32)
- `--learning-rate` - Learning rate for RL training (default: 1e-3)

### Model Persistence
- `--load-model` - Path to load pre-trained model
- `--save-model` - Path to save trained model

### Misc
- `--seed` - Random seed for reproducibility (default: 42)
- `--plot` - Generate and save fitness evolution plots

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

**When to Use Variable-Length DNC (`dnc-gp`):**
- GP problems where you want trainable crossover for trees
- Automatically handles padding and validation
- Can be combined with BERT mutation using shared tokenizer
- Falls back to parent if offspring is invalid

**Key Differences from Standard BERT:**
- No ground-truth labels → uses REINFORCE for training
- Sequential replacement in DFS order (not simultaneous)
- Type constraints (terminals vs functions, arity matching)
- Reward = fitness improvement (not cross-entropy loss)

**Unified Tokenization:**
- When using both `dnc-gp` crossover and `bert` mutation, they share the same tokenizer instance
- This prevents vocabulary inconsistencies and improves interoperability
- Tokenizer is created once and passed to both operators

## Code Origins

Base GA implementation adapted from [github.com/Mruzik1/Genetic-Algorithms](https://github.com/Mruzik1/Genetic-Algorithms/tree/main). DNC operator, Variable-Length DNC, BERT mutation operator, and RL training infrastructure implemented for this assignment.
