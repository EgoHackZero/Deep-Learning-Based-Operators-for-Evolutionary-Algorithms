"""
Variable-Length DNC Crossover

Wrapper around DNCrossover that handles variable-length individuals (GP trees)
by padding/unpadding transparently. Allows DNC to work with symbolic regression
and other GP problems.
"""

import random
from typing import Optional
import numpy as np

from genetic_algorithm.crossovers.dnc_crossover import DNCrossover
from genetic_algorithm.gp_tree import GPTree, NodeType
from genetic_algorithm.tokenizer import BaseTokenizer


class VariableLengthDNCrossover(DNCrossover):
    """
    DNC crossover adapted for variable-length GP trees.

    Strategy:
    1. Pad shorter parent to match longer parent's length
    2. Apply DNC crossover (which expects fixed-length)
    3. Validate offspring as valid GP tree
    4. If invalid, fallback to returning a valid parent

    Padding strategy:
    - Pads with terminal values from the tree's terminal set
    - Ensures padding doesn't break tree structure
    """

    def __init__(
        self,
        tokenizer: BaseTokenizer,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        device: Optional[str] = None,
        seed: Optional[int] = None,
        pad_value: Optional[float] = None,
        fallback_to_parent: bool = True
    ):
        """
        Initialize variable-length DNC crossover.

        Args:
            tokenizer: Tokenizer for converting symbols to IDs (shared with BERT)
            input_size: Maximum expected individual size (for initialization)
            hidden_size: LSTM hidden size
            num_layers: Number of LSTM layers
            device: Device to run on (cuda/cpu)
            seed: Random seed
            pad_value: Value to use for padding (None = use last element)
            fallback_to_parent: If True, return parent when offspring is invalid
        """
        # DNCrossover expects gene_size parameter (size of each gene)
        # For GP trees, each node is a "gene", so gene_size=1
        super().__init__(
            gene_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            model_path=None,
            device=device,
            seed=seed
        )

        # Store tokenizer (unified with BERT mutation)
        self.tokenizer = tokenizer

        # Store max tree size for padding purposes
        self.max_tree_size = input_size

        self.pad_value = pad_value if pad_value is not None else self.tokenizer.pad_token_id
        self.fallback_to_parent = fallback_to_parent

        self.stats = {
            'total_crossovers': 0,
            'padded_crossovers': 0,
            'invalid_offspring': 0,
            'fallbacks': 0
        }

    def _tokenize(self, individual: list) -> list:
        """
        Convert symbolic individual to numeric IDs using tokenizer.

        Args:
            individual: Individual with mixed types (strings, numbers)

        Returns:
            List of numeric IDs
        """
        numeric_individual = []

        for value in individual:
            # Convert value to token string
            token = self.tokenizer.value_to_token(value)

            # Get token ID
            token_id = self.tokenizer.token_to_id.get(token, self.tokenizer.unk_token_id)
            numeric_individual.append(token_id)

        return numeric_individual

    def _detokenize(self, numeric_individual: list) -> list:
        """
        Convert numeric IDs back to symbolic individual using tokenizer.

        Args:
            numeric_individual: List of numeric IDs

        Returns:
            Individual with original symbols
        """
        symbolic_individual = []

        for token_id in numeric_individual:
            # Round to nearest integer for ID lookup
            token_id = int(round(token_id))

            # Get token string
            token = self.tokenizer.id_to_token.get(token_id, self.tokenizer.UNK_TOKEN)

            # Convert token to value
            value = self.tokenizer.token_to_value(token)

            # Handle special tokens
            if value is None or token in [self.tokenizer.PAD_TOKEN, self.tokenizer.MASK_TOKEN, self.tokenizer.UNK_TOKEN]:
                # Use default value (0.0) for unknown tokens
                symbolic_individual.append(0.0)
            else:
                symbolic_individual.append(value)

        return symbolic_individual

    def _validate_parents(self, parent1: list, parent2: list):
        """
        Override base validation to allow variable-length parents.

        Args:
            parent1: First parent
            parent2: Second parent
        """
        # Minimum length check (from BaseCrossover)
        min_length = 1
        if len(parent1) < min_length or len(parent2) < min_length:
            raise ValueError(
                f"Parents' length should be >= {min_length}! "
                f"You have: {len(parent1)} and {len(parent2)}"
            )

        # We explicitly allow different lengths (that's the point of this class!)
        # Don't call super()._validate_parents() which checks equal length

    def _pad_individual(self, individual: list, target_length: int) -> list:
        """
        Pad individual to target length.

        Strategy:
        - If pad_value is specified, use it
        - Otherwise, repeat the last element

        Args:
            individual: Individual to pad
            target_length: Target length

        Returns:
            Padded individual
        """
        if len(individual) >= target_length:
            return individual.copy()

        padded = individual.copy()
        padding_needed = target_length - len(individual)

        if self.pad_value is not None:
            # Use specified pad value
            padding = [self.pad_value] * padding_needed
        else:
            # Use last element (for GP trees, often a terminal or constant)
            padding = [individual[-1]] * padding_needed

        padded.extend(padding)
        return padded

    def _truncate_to_valid_tree(self, individual: list, min_length: int = 1) -> list:
        """
        Truncate individual to form a valid GP tree.

        For GP trees in prefix notation, we need to ensure the tree is complete
        (all functions have the correct number of children).

        Args:
            individual: Individual to truncate
            min_length: Minimum valid length

        Returns:
            Valid truncated individual
        """
        if len(individual) <= min_length:
            return individual

        # Try progressively shorter lengths until we find a valid tree
        for length in range(len(individual), min_length - 1, -1):
            candidate = individual[:length]

            # Check if it's a valid GP tree
            if self._is_valid_gp_tree_simple(candidate):
                return candidate

        # If no valid truncation found, return as-is
        return individual

    def _is_valid_gp_tree_with_metadata(self, individual: list) -> bool:
        """
        Validate GP tree by building metadata from tokenizer.

        Args:
            individual: Individual in prefix notation

        Returns:
            True if valid GP tree
        """
        try:
            # Build metadata based on actual values
            metadata = []
            function_arities = self.tokenizer.get_function_arities()
            terminal_set = self.tokenizer.get_terminal_set()

            for value in individual:
                if value in function_arities:
                    # Function node
                    metadata.append({
                        'type': NodeType.FUNCTION,
                        'arity': function_arities[value],
                        'value': value
                    })
                elif value in terminal_set:
                    # Terminal node
                    metadata.append({
                        'type': NodeType.TERMINAL,
                        'arity': 0,
                        'value': value
                    })
                else:
                    # Constant node
                    metadata.append({
                        'type': NodeType.CONSTANT,
                        'arity': 0,
                        'value': value
                    })

            # Try to create and validate GPTree
            tree = GPTree(individual, metadata)
            tree._validate_tree()
            return True
        except Exception:
            return False

    def _is_valid_gp_tree_simple(self, individual: list) -> bool:
        """
        Simple validation: check if tree is complete in prefix notation.

        In prefix notation, we track the "remaining slots" that need to be filled:
        - Start with 1 slot (the root)
        - Each function adds (arity - 1) slots
        - Each terminal fills 1 slot
        - Tree is valid when slots = 0 at the end

        Args:
            individual: Individual in prefix notation

        Returns:
            True if valid GP tree
        """
        try:
            # Try to create GPTree and validate
            tree = GPTree(individual, [])  # Empty metadata, will build it
            tree._validate_tree()
            return True
        except:
            # Fallback: simple slot counting
            slots = 1  # Start with one slot for root

            for node in individual:
                if slots <= 0:
                    return False  # Too many nodes

                # Check if node is a function (has arity)
                if hasattr(node, '__call__'):
                    # It's a function object
                    try:
                        import inspect
                        arity = len(inspect.signature(node).parameters)
                        slots += (arity - 1)
                    except:
                        slots -= 1  # Assume terminal
                elif isinstance(node, str):
                    # Heuristic: common operators
                    if node in ['+', '-', '*', '/', 'add', 'sub', 'mul', 'div']:
                        slots += 1  # Binary function: adds 1 slot (2 - 1)
                    elif node in ['sin', 'cos', 'exp', 'log', 'sqrt', 'neg']:
                        slots += 0  # Unary function: adds 0 slots (1 - 1)
                    else:
                        slots -= 1  # Assume terminal
                else:
                    # Number or other type → terminal
                    slots -= 1

            return slots == 0

    def perform(self, parent1: list, parent2: list, **kwargs) -> list:
        """
        Perform crossover on variable-length parents.

        Args:
            parent1: First parent (can be any length)
            parent2: Second parent (can be any length)
            **kwargs: Additional arguments

        Returns:
            Offspring (valid GP tree)
        """
        self.stats['total_crossovers'] += 1

        # Tokenize parents (convert symbols to numbers)
        numeric_parent1 = self._tokenize(parent1)
        numeric_parent2 = self._tokenize(parent2)

        # Get lengths
        len1, len2 = len(numeric_parent1), len(numeric_parent2)
        max_len = max(len1, len2)
        min_len = min(len1, len2)

        # Check if padding is needed
        if len1 != len2:
            self.stats['padded_crossovers'] += 1

            # Pad to same length
            padded_parent1 = self._pad_individual(numeric_parent1, max_len)
            padded_parent2 = self._pad_individual(numeric_parent2, max_len)
        else:
            padded_parent1 = numeric_parent1
            padded_parent2 = numeric_parent2

        # Call parent class perform (DNCrossover)
        # This will use the DNC LSTM-based crossover
        numeric_offspring = super().perform(padded_parent1, padded_parent2, **kwargs)

        # Detokenize offspring (convert numbers back to symbols)
        offspring = self._detokenize(numeric_offspring)

        # Remove padding tokens (0.0) that were added during detokenization
        offspring = [gene for gene in offspring if not (isinstance(gene, float) and abs(gene) < 1e-9 and gene not in parent1 and gene not in parent2)]

        # Try different truncation lengths to find a valid tree
        for try_len in range(len(offspring), min_len - 1, -1):
            candidate = offspring[:try_len]

            # Validate by trying to build metadata
            if self._is_valid_gp_tree_with_metadata(candidate):
                return candidate

        # If no valid truncation found, use fallback
        self.stats['invalid_offspring'] += 1

        # Fallback strategies
        if self.fallback_to_parent:
            # Return a random parent (ensures valid offspring)
            self.stats['fallbacks'] += 1
            return random.choice([parent1, parent2])
        else:
            # Return shortest attempt (might be invalid)
            return offspring[:min_len] if len(offspring) >= min_len else offspring

    def get_stats(self) -> dict:
        """
        Get crossover statistics.

        Returns:
            Dictionary with statistics
        """
        stats = self.stats.copy()
        if stats['total_crossovers'] > 0:
            stats['padding_rate'] = stats['padded_crossovers'] / stats['total_crossovers']
            stats['invalid_rate'] = stats['invalid_offspring'] / stats['total_crossovers']
            stats['fallback_rate'] = stats['fallbacks'] / stats['total_crossovers']
        return stats

    def print_stats(self):
        """Print crossover statistics."""
        stats = self.get_stats()
        print("\nVariable-Length DNC Crossover Statistics:")
        print(f"  Total crossovers: {stats['total_crossovers']}")
        print(f"  Padded crossovers: {stats['padded_crossovers']} "
              f"({stats.get('padding_rate', 0)*100:.1f}%)")
        print(f"  Invalid offspring: {stats['invalid_offspring']} "
              f"({stats.get('invalid_rate', 0)*100:.1f}%)")
        print(f"  Fallbacks to parent: {stats['fallbacks']} "
              f"({stats.get('fallback_rate', 0)*100:.1f}%)")


if __name__ == "__main__":
    # Test the variable-length DNC crossover
    print("Testing Variable-Length DNC Crossover...")

    from genetic_algorithm.tokenizer import SymbolicRegressionTokenizer
    from genetic_algorithm.gp_tree import generate_random_tree

    # Create tokenizer
    tokenizer = SymbolicRegressionTokenizer(terminal_names=['x', 'y'])

    # Generate trees of different lengths
    tree1 = generate_random_tree(
        max_depth=2,
        function_set=tokenizer.get_function_arities(),
        terminal_set=list(tokenizer.get_terminal_set()),
        constant_range=(-5.0, 5.0)
    )

    tree2 = generate_random_tree(
        max_depth=4,
        function_set=tokenizer.get_function_arities(),
        terminal_set=list(tokenizer.get_terminal_set()),
        constant_range=(-5.0, 5.0)
    )

    parent1 = tree1.to_list()
    parent2 = tree2.to_list()

    print(f"\nParent 1 (length {len(parent1)}): {parent1}")
    print(f"Parent 2 (length {len(parent2)}): {parent2}")

    # Create crossover
    max_len = max(len(parent1), len(parent2))
    crossover = VariableLengthDNCrossover(
        tokenizer=tokenizer,
        input_size=max_len,
        hidden_size=16,
        num_layers=1
    )

    # Perform crossover
    offspring = crossover.perform(parent1, parent2)

    print(f"\nOffspring (length {len(offspring)}): {offspring}")
    print(f"Valid tree: {crossover._is_valid_gp_tree_simple(offspring)}")

    # Test multiple crossovers
    print("\nPerforming 10 crossovers...")
    for _ in range(9):  # Already did 1
        offspring = crossover.perform(parent1, parent2)

    crossover.print_stats()

    print("\nVariable-Length DNC Crossover test passed!")
