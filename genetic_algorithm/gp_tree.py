"""
GP Tree Representation Utilities

Provides utilities for representing genetic programming trees in prefix notation
and converting between tree structures and flat list representations.
"""

from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum


class NodeType(Enum):
    """Types of nodes in a GP tree"""
    FUNCTION = "function"
    TERMINAL = "terminal"
    CONSTANT = "constant"


@dataclass
class Node:
    """Represents a node in a GP tree"""
    value: Any  # The actual value (operator, variable, or constant)
    node_type: NodeType
    arity: int = 0  # Number of children (0 for terminals/constants)

    def __str__(self):
        return str(self.value)

    def __repr__(self):
        return f"Node({self.value}, {self.node_type.value}, arity={self.arity})"


class GPTree:
    """
    Represents a genetic programming tree using prefix notation.

    Example:
        Expression: (x + y) * 2
        Prefix notation: ['*', '+', 'x', 'y', '2']
        Tree structure:
               *
              / \
             +   2
            / \
           x   y
    """

    def __init__(self, prefix_sequence: List[Any], node_metadata: Optional[List[Dict]] = None):
        """
        Initialize GP tree from prefix notation sequence.

        Args:
            prefix_sequence: List representing the tree in prefix order
            node_metadata: Optional list of metadata dicts for each node
                          (contains 'type', 'arity' for each position)
        """
        self.prefix_sequence = prefix_sequence
        self.node_metadata = node_metadata or []

        # Validate tree structure if metadata is provided
        if self.node_metadata:
            self._validate_tree()

    def _validate_tree(self) -> bool:
        """
        Validates that the prefix sequence forms a valid tree.
        Uses stack-based validation: each function consumes its arity children.

        Returns:
            True if valid

        Raises:
            ValueError if tree structure is invalid
        """
        if len(self.prefix_sequence) != len(self.node_metadata):
            raise ValueError(
                f"Sequence length {len(self.prefix_sequence)} doesn't match "
                f"metadata length {len(self.node_metadata)}"
            )

        # Stack tracks how many children each node still needs
        needed_children = 0

        for i, (node, meta) in enumerate(zip(self.prefix_sequence, self.node_metadata)):
            arity = meta.get('arity', 0)

            if needed_children == 0 and i > 0:
                raise ValueError(f"Tree completed before end at position {i}")

            # This node satisfies one child requirement from parent
            if i > 0:
                needed_children -= 1

            # This node adds its own children requirements
            needed_children += arity

        if needed_children != 0:
            raise ValueError(f"Incomplete tree: still need {needed_children} children")

        return True

    def to_list(self) -> List[Any]:
        """Returns the flat prefix sequence representation"""
        return self.prefix_sequence.copy()

    def __len__(self) -> int:
        """Returns the number of nodes in the tree"""
        return len(self.prefix_sequence)

    def __str__(self) -> str:
        """String representation in prefix notation"""
        return ' '.join(str(x) for x in self.prefix_sequence)

    def __getitem__(self, index: int) -> Any:
        """Access nodes by index"""
        return self.prefix_sequence[index]

    def __setitem__(self, index: int, value: Any):
        """Modify nodes by index"""
        self.prefix_sequence[index] = value

    @classmethod
    def from_list(cls, flat_list: List[Any], node_metadata: Optional[List[Dict]] = None) -> 'GPTree':
        """
        Creates a GPTree from a flat list representation.

        Args:
            flat_list: Flat list in prefix notation
            node_metadata: Optional metadata for each node

        Returns:
            GPTree instance
        """
        return cls(flat_list.copy(), node_metadata)

    def get_metadata(self, index: int) -> Dict:
        """
        Get metadata for a specific node.

        Args:
            index: Position in the prefix sequence

        Returns:
            Metadata dictionary with 'type' and 'arity'
        """
        if not self.node_metadata or index >= len(self.node_metadata):
            return {'type': NodeType.TERMINAL, 'arity': 0}
        return self.node_metadata[index]

    def get_dfs_order(self) -> List[int]:
        """
        Returns indices in depth-first search order.
        This is the order used for sequential mask replacement in BERT mutation.

        Returns:
            List of indices in DFS order (parent before children)
        """
        if not self.node_metadata:
            # Without metadata, return sequential order
            return list(range(len(self.prefix_sequence)))

        dfs_order = []
        self._dfs_traverse(0, dfs_order)
        return dfs_order

    def _dfs_traverse(self, index: int, order: List[int]) -> int:
        """
        Helper for DFS traversal.

        Args:
            index: Current node index
            order: List to append indices to

        Returns:
            Index after processing this subtree
        """
        if index >= len(self.prefix_sequence):
            return index

        # Visit current node
        order.append(index)

        # Get arity of current node
        arity = self.node_metadata[index].get('arity', 0)

        # Visit children left to right
        child_index = index + 1
        for _ in range(arity):
            child_index = self._dfs_traverse(child_index, order)

        return child_index


def build_tree_from_expression(expression: List[Any],
                               function_arities: Dict[str, int],
                               terminal_set: set) -> GPTree:
    """
    Builds a GPTree with proper metadata from an expression.

    Args:
        expression: List representing the tree in prefix notation
        function_arities: Dict mapping function names to their arities
        terminal_set: Set of terminal symbols (variables)

    Returns:
        GPTree with metadata

    Example:
        expression = ['*', '+', 'x', 'y', 2.5]
        function_arities = {'+': 2, '*': 2}
        terminal_set = {'x', 'y'}

        Returns GPTree with proper node types and arities
    """
    metadata = []

    for token in expression:
        if token in function_arities:
            # Function node
            metadata.append({
                'type': NodeType.FUNCTION,
                'arity': function_arities[token],
                'value': token
            })
        elif token in terminal_set:
            # Terminal (variable) node
            metadata.append({
                'type': NodeType.TERMINAL,
                'arity': 0,
                'value': token
            })
        else:
            # Constant node
            metadata.append({
                'type': NodeType.CONSTANT,
                'arity': 0,
                'value': token
            })

    return GPTree(expression, metadata)


def generate_random_tree(max_depth: int,
                        function_set: Dict[str, int],
                        terminal_set: List[str],
                        constant_range: Tuple[float, float] = (-10.0, 10.0),
                        prob_terminal: float = 0.3) -> GPTree:
    """
    Generates a random GP tree using the grow method.

    Args:
        max_depth: Maximum depth of the tree
        function_set: Dict mapping function names to arities
        terminal_set: List of terminal symbols
        constant_range: Range for generating random constants
        prob_terminal: Probability of choosing terminal at non-leaf nodes

    Returns:
        Random GPTree
    """
    import random

    def grow(depth: int) -> Tuple[List[Any], List[Dict]]:
        """Recursively grow a tree"""
        # At max depth or randomly choose terminal
        if depth >= max_depth or (depth > 0 and random.random() < prob_terminal):
            # Choose terminal or constant
            if random.random() < 0.7 and terminal_set:
                # Terminal
                term = random.choice(terminal_set)
                return [term], [{'type': NodeType.TERMINAL, 'arity': 0, 'value': term}]
            else:
                # Constant
                const = random.uniform(*constant_range)
                return [const], [{'type': NodeType.CONSTANT, 'arity': 0, 'value': const}]
        else:
            # Choose function
            func = random.choice(list(function_set.keys()))
            arity = function_set[func]

            sequence = [func]
            metadata = [{'type': NodeType.FUNCTION, 'arity': arity, 'value': func}]

            # Generate children
            for _ in range(arity):
                child_seq, child_meta = grow(depth + 1)
                sequence.extend(child_seq)
                metadata.extend(child_meta)

            return sequence, metadata

    sequence, metadata = grow(0)
    return GPTree(sequence, metadata)


if __name__ == "__main__":
    # Test the GP tree implementation
    print("Testing GP Tree implementation...")

    # Example: (x + y) * 2
    expression = ['*', '+', 'x', 'y', 2.0]
    function_arities = {'+': 2, '*': 2}
    terminal_set = {'x', 'y'}

    tree = build_tree_from_expression(expression, function_arities, terminal_set)
    print(f"\nTree: {tree}")
    print(f"Length: {len(tree)}")
    print(f"Valid: {tree._validate_tree()}")

    # Test DFS order
    dfs = tree.get_dfs_order()
    print(f"\nDFS order indices: {dfs}")
    print("DFS order nodes:", [tree[i] for i in dfs])

    # Generate random tree
    print("\nGenerating random tree...")
    random_tree = generate_random_tree(
        max_depth=3,
        function_set={'+': 2, '-': 2, '*': 2, '/': 2},
        terminal_set=['x', 'y', 'z'],
        constant_range=(-5.0, 5.0)
    )
    print(f"Random tree: {random_tree}")
    print(f"Length: {len(random_tree)}")
