"""
Symbolic Regression Problem

Genetic programming problem to evolve mathematical expressions that fit target data.
"""

import numpy as np
from typing import List, Callable, Optional, Tuple
from problems.base_problem import BaseProblem
from genetic_algorithm.gp_tree import generate_random_tree, GPTree


class SymbolicRegression(BaseProblem):
    """
    Symbolic regression using genetic programming.

    Goal: Evolve a mathematical expression that fits the target data points.
    """

    def __init__(
        self,
        target_function: Optional[Callable] = None,
        x_range: Tuple[float, float] = (-10.0, 10.0),
        num_points: int = 50,
        max_tree_depth: int = 5,
        function_set: Optional[dict] = None,
        terminal_set: Optional[List[str]] = None,
        constant_range: Tuple[float, float] = (-10.0, 10.0),
        penalty_coefficient: float = 0.001
    ):
        """
        Initialize symbolic regression problem.

        Args:
            target_function: Function to approximate (default: x^2 + x + 1)
            x_range: Range of x values to sample
            num_points: Number of data points to fit
            max_tree_depth: Maximum depth for generated trees
            function_set: Dict of {function_name: arity}
            terminal_set: List of terminal symbols (variables)
            constant_range: Range for random constants
            penalty_coefficient: Coefficient for tree size penalty
        """
        # Default target: x^2 + x + 1
        if target_function is None:
            target_function = lambda x: x**2 + x + 1

        self.target_function = target_function
        self.max_tree_depth = max_tree_depth
        self.constant_range = constant_range
        self.penalty_coefficient = penalty_coefficient

        # Generate target data
        self.x_data = np.linspace(x_range[0], x_range[1], num_points)
        self.y_data = np.array([target_function(x) for x in self.x_data])

        # Default function set
        if function_set is None:
            self.function_set = {
                '+': 2,
                '-': 2,
                '*': 2,
                '/': 2  # Protected division
            }
        else:
            self.function_set = function_set

        # Default terminal set
        if terminal_set is None:
            self.terminal_set = ['x']
        else:
            self.terminal_set = terminal_set

    def fitness_function(self, individual: list) -> float:
        """
        Evaluate fitness of a GP tree.

        Fitness = -MSE - penalty * tree_size
        (Higher is better, so we negate MSE)

        Args:
            individual: GP tree in prefix notation (flat list)

        Returns:
            Fitness score (higher is better)
        """
        try:
            # Evaluate tree on all x points
            predictions = []
            for x in self.x_data:
                value = self._evaluate_tree(individual, {'x': x})

                # Check for invalid values
                if not np.isfinite(value):
                    return -1e6  # Large penalty for invalid expressions

                predictions.append(value)

            # Compute mean squared error
            mse = np.mean((np.array(predictions) - self.y_data) ** 2)

            # Add size penalty (encourage smaller trees)
            size_penalty = self.penalty_coefficient * len(individual)

            # Fitness = negative error (higher is better)
            fitness = -mse - size_penalty

            return fitness

        except Exception:
            # If evaluation fails, return very low fitness
            return -1e6

    def _evaluate_tree(self, tree: list, variables: dict) -> float:
        """
        Evaluate a GP tree recursively.

        Args:
            tree: GP tree in prefix notation
            variables: Dictionary mapping variable names to values

        Returns:
            Evaluated value
        """
        if not tree:
            return 0.0

        # Use a stack-based approach to avoid recursion issues
        stack = []
        index = len(tree) - 1

        def eval_from_index(idx):
            if idx < 0:
                return 0.0, -1

            node = tree[idx]

            # Check if it's a function
            if node in self.function_set:
                arity = self.function_set[node]
                args = []

                # Evaluate children from right to left (since we're going backwards)
                current_idx = idx - 1
                for _ in range(arity):
                    arg_value, current_idx = eval_from_index(current_idx)
                    args.insert(0, arg_value)  # Insert at beginning

                # Apply function
                result = self._apply_function(node, args)
                return result, current_idx

            # Check if it's a terminal
            elif node in variables:
                return variables[node], idx - 1

            # Otherwise it's a constant
            else:
                try:
                    return float(node), idx - 1
                except (ValueError, TypeError):
                    return 0.0, idx - 1

        result, _ = eval_from_index(len(tree) - 1)
        return result

    def _apply_function(self, func: str, args: List[float]) -> float:
        """
        Apply a function to arguments.

        Args:
            func: Function name
            args: List of arguments

        Returns:
            Result value
        """
        if func == '+':
            return args[0] + args[1]
        elif func == '-':
            return args[0] - args[1]
        elif func == '*':
            return args[0] * args[1]
        elif func == '/':
            # Protected division
            if abs(args[1]) < 1e-6:
                return 1.0
            return args[0] / args[1]
        elif func == 'sin':
            return np.sin(args[0])
        elif func == 'cos':
            return np.cos(args[0])
        elif func == 'exp':
            # Clip to prevent overflow
            return np.exp(np.clip(args[0], -10, 10))
        elif func == 'log':
            # Protected log
            return np.log(abs(args[0]) + 1e-6)
        elif func == 'pow':
            # Protected power
            try:
                result = np.power(abs(args[0]), abs(args[1]) % 3)  # Limit exponent
                if np.isfinite(result):
                    return result
                return 1.0
            except:
                return 1.0
        elif func == 'sqrt':
            return np.sqrt(abs(args[0]))
        else:
            return 0.0

    def generate_individual(self) -> list:
        """
        Generate a random GP tree.

        Returns:
            GP tree in prefix notation (flat list)
        """
        tree = generate_random_tree(
            max_depth=self.max_tree_depth,
            function_set=self.function_set,
            terminal_set=self.terminal_set,
            constant_range=self.constant_range,
            prob_terminal=0.3
        )
        return tree.to_list()

    def problem_name(self) -> str:
        """Return problem name"""
        return "symbolic_regression"

    def get_function_arities(self) -> dict:
        """Get function set with arities"""
        return self.function_set.copy()

    def get_terminal_set(self) -> List[str]:
        """Get terminal set"""
        return self.terminal_set.copy()

    def get_expression_string(self, individual: list) -> str:
        """
        Convert GP tree to readable expression string.

        Args:
            individual: GP tree in prefix notation

        Returns:
            Human-readable expression
        """
        def to_infix(tree, idx=0):
            if idx >= len(tree):
                return "", idx

            node = tree[idx]

            # Function node
            if node in self.function_set:
                arity = self.function_set[node]
                idx += 1

                if arity == 1:
                    # Unary function
                    arg, idx = to_infix(tree, idx)
                    return f"{node}({arg})", idx
                elif arity == 2:
                    # Binary function
                    left, idx = to_infix(tree, idx)
                    right, idx = to_infix(tree, idx)

                    if node in ['+', '-', '*', '/']:
                        return f"({left} {node} {right})", idx
                    else:
                        return f"{node}({left}, {right})", idx

            # Terminal or constant
            elif isinstance(node, (int, float)):
                return f"{node:.2f}", idx + 1
            else:
                return str(node), idx + 1

        expr, _ = to_infix(individual)
        return expr


if __name__ == "__main__":
    # Test symbolic regression
    print("Testing Symbolic Regression Problem...")

    # Create problem: target function x^2
    problem = SymbolicRegression(
        target_function=lambda x: x**2,
        x_range=(-5.0, 5.0),
        num_points=20,
        max_tree_depth=3
    )

    print(f"Problem: {problem.problem_name()}")
    print(f"Data points: {len(problem.x_data)}")
    print(f"Function set: {problem.function_set}")
    print(f"Terminal set: {problem.terminal_set}")

    # Generate and evaluate individuals
    print("\nGenerating random individuals...")
    for i in range(3):
        individual = problem.generate_individual()
        fitness = problem.fitness_function(individual)
        expr = problem.get_expression_string(individual)

        print(f"\nIndividual {i+1}:")
        print(f"  Prefix: {individual}")
        print(f"  Expression: {expr}")
        print(f"  Fitness: {fitness:.4f}")

    # Test perfect solution: [x, x, *] = x * x = x^2
    perfect = ['*', 'x', 'x']
    fitness = problem.fitness_function(perfect)
    expr = problem.get_expression_string(perfect)
    print(f"\nPerfect solution test:")
    print(f"  Prefix: {perfect}")
    print(f"  Expression: {expr}")
    print(f"  Fitness: {fitness:.4f}")

    print("\nSymbolic Regression test passed!")
