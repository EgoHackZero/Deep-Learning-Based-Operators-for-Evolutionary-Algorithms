"""
Tokenization utilities for BERT mutation operator.

Handles conversion between GP trees and token sequences for the BERT model.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Any, Optional
import torch
from genetic_algorithm.gp_tree import GPTree, NodeType


class BaseTokenizer(ABC):
    """
    Abstract base class for tokenizers.

    Each problem domain needs its own tokenizer to handle problem-specific
    operators, terminals, and constants.
    """

    # Special tokens
    PAD_TOKEN = "[PAD]"
    MASK_TOKEN = "[MASK]"
    UNK_TOKEN = "[UNK]"

    def __init__(self):
        """Initialize tokenizer with vocabulary"""
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        self.vocab_size = 0

        # Initialize special tokens
        self._add_special_tokens()

        # Build vocabulary
        self._build_vocab()

    def _add_special_tokens(self):
        """Add special tokens to vocabulary"""
        self.pad_token_id = self._add_token(self.PAD_TOKEN)
        self.mask_token_id = self._add_token(self.MASK_TOKEN)
        self.unk_token_id = self._add_token(self.UNK_TOKEN)

    def _add_token(self, token: str) -> int:
        """
        Add a token to the vocabulary.

        Args:
            token: Token string to add

        Returns:
            Token ID
        """
        if token not in self.token_to_id:
            token_id = len(self.token_to_id)
            self.token_to_id[token] = token_id
            self.id_to_token[token_id] = token
            self.vocab_size += 1
            return token_id
        return self.token_to_id[token]

    @abstractmethod
    def _build_vocab(self):
        """Build problem-specific vocabulary. Must be implemented by subclasses."""
        ...

    @abstractmethod
    def get_function_arities(self) -> Dict[str, int]:
        """Return dictionary mapping function names to their arities"""
        ...

    @abstractmethod
    def get_terminal_set(self) -> set:
        """Return set of terminal symbols"""
        ...

    @abstractmethod
    def value_to_token(self, value: Any) -> str:
        """Convert a node value to a token string"""
        ...

    @abstractmethod
    def token_to_value(self, token: str) -> Any:
        """Convert a token string back to a node value"""
        ...

    def encode(self, tree: GPTree) -> Tuple[torch.Tensor, List[Dict]]:
        """
        Encode a GP tree into token IDs.

        Args:
            tree: GPTree to encode

        Returns:
            Tuple of (token_ids tensor, metadata list)
            - token_ids: LongTensor of shape [seq_len]
            - metadata: List of dicts with node type and arity info
        """
        token_ids = []
        metadata = []

        for i, value in enumerate(tree.prefix_sequence):
            # Convert value to token string
            token = self.value_to_token(value)

            # Get token ID (use UNK if not in vocab)
            token_id = self.token_to_id.get(token, self.unk_token_id)
            token_ids.append(token_id)

            # Get metadata
            meta = tree.get_metadata(i)
            metadata.append(meta)

        return torch.tensor(token_ids, dtype=torch.long), metadata

    def decode(self, token_ids: torch.Tensor, metadata: List[Dict]) -> GPTree:
        """
        Decode token IDs back into a GP tree.

        Args:
            token_ids: Tensor of token IDs [seq_len]
            metadata: List of metadata dicts

        Returns:
            GPTree instance
        """
        sequence = []

        for token_id in token_ids.tolist():
            # Convert ID to token string
            token = self.id_to_token.get(token_id, self.UNK_TOKEN)

            # Skip special tokens
            if token in [self.PAD_TOKEN, self.MASK_TOKEN, self.UNK_TOKEN]:
                continue

            # Convert token to value
            value = self.token_to_value(token)
            sequence.append(value)

        return GPTree(sequence, metadata)

    def get_valid_replacements(self, node_type: NodeType, arity: Optional[int] = None) -> List[int]:
        """
        Get valid token IDs for replacing a masked node.

        Args:
            node_type: Type of the node being replaced
            arity: Required arity for function nodes (None for terminals/constants)

        Returns:
            List of valid token IDs
        """
        valid_ids = []

        for token, token_id in self.token_to_id.items():
            # Skip special tokens
            if token in [self.PAD_TOKEN, self.MASK_TOKEN, self.UNK_TOKEN]:
                continue

            # Check if token matches the required type
            if node_type == NodeType.FUNCTION:
                # For functions, check arity
                func_arities = self.get_function_arities()
                if token in func_arities and (arity is None or func_arities[token] == arity):
                    valid_ids.append(token_id)

            elif node_type == NodeType.TERMINAL:
                # For terminals, check if in terminal set
                if token in self.get_terminal_set():
                    valid_ids.append(token_id)

            elif node_type == NodeType.CONSTANT:
                # For constants, use special constant token
                if token.startswith("CONST_"):
                    valid_ids.append(token_id)

        return valid_ids

    def pad_sequence(self, token_ids: torch.Tensor, max_length: int) -> torch.Tensor:
        """
        Pad token sequence to max_length.

        Args:
            token_ids: Token IDs tensor [seq_len]
            max_length: Target length

        Returns:
            Padded tensor [max_length]
        """
        if len(token_ids) >= max_length:
            return token_ids[:max_length]

        padding = torch.full((max_length - len(token_ids),), self.pad_token_id, dtype=torch.long)
        return torch.cat([token_ids, padding])

    def create_attention_mask(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Create attention mask (1 for real tokens, 0 for padding).

        Args:
            token_ids: Token IDs tensor [seq_len]

        Returns:
            Attention mask tensor [seq_len]
        """
        return (token_ids != self.pad_token_id).long()


class SymbolicRegressionTokenizer(BaseTokenizer):
    """
    Tokenizer for symbolic regression problems.

    Vocabulary includes:
    - Special tokens: [PAD], [MASK], [UNK]
    - Functions: +, -, *, /, sin, cos, exp, log
    - Terminals: x (can be extended)
    - Constants: CONST_TOKEN (placeholder, actual value stored separately)
    """

    def __init__(self, terminal_names: Optional[List[str]] = None):
        """
        Initialize symbolic regression tokenizer.

        Args:
            terminal_names: List of terminal variable names (default: ['x'])
        """
        self.terminal_names = terminal_names or ['x']
        self._function_arities = {
            '+': 2,
            '-': 2,
            '*': 2,
            '/': 2,
            'sin': 1,
            'cos': 1,
            'exp': 1,
            'log': 1,
            'pow': 2,
            'sqrt': 1
        }
        super().__init__()

    def _build_vocab(self):
        """Build vocabulary for symbolic regression"""
        # Add functions
        for func in self._function_arities.keys():
            self._add_token(func)

        # Add terminals
        for term in self.terminal_names:
            self._add_token(term)

        # Add constant placeholder
        self._add_token("CONST_TOKEN")

    def get_function_arities(self) -> Dict[str, int]:
        """Return function arities"""
        return self._function_arities.copy()

    def get_terminal_set(self) -> set:
        """Return terminal set"""
        return set(self.terminal_names)

    def value_to_token(self, value: Any) -> str:
        """
        Convert node value to token string.

        Args:
            value: Node value (operator, terminal, or constant)

        Returns:
            Token string
        """
        # Check if it's a function
        if value in self._function_arities:
            return str(value)

        # Check if it's a terminal
        if value in self.terminal_names:
            return str(value)

        # Otherwise treat as constant
        # All constants map to CONST_TOKEN
        return "CONST_TOKEN"

    def token_to_value(self, token: str) -> Any:
        """
        Convert token back to value.

        Args:
            token: Token string

        Returns:
            Node value
        """
        # Check if it's a function or terminal
        if token in self._function_arities or token in self.terminal_names:
            return token

        # For constants, return None (caller needs to handle)
        if token == "CONST_TOKEN":
            return None

        # Unknown token
        return None


if __name__ == "__main__":
    # Test tokenizer
    print("Testing Symbolic Regression Tokenizer...")

    from genetic_algorithm.gp_tree import build_tree_from_expression

    tokenizer = SymbolicRegressionTokenizer(terminal_names=['x', 'y'])
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print(f"Token to ID: {tokenizer.token_to_id}")

    # Create a simple tree: (x + y) * 2
    expression = ['*', '+', 'x', 'y', 2.5]
    tree = build_tree_from_expression(
        expression,
        tokenizer.get_function_arities(),
        tokenizer.get_terminal_set()
    )

    # Encode
    token_ids, metadata = tokenizer.encode(tree)
    print(f"\nOriginal tree: {tree}")
    print(f"Token IDs: {token_ids}")
    print(f"Metadata: {metadata}")

    # Test valid replacements
    print("\nValid replacements for FUNCTION with arity 2:")
    valid = tokenizer.get_valid_replacements(NodeType.FUNCTION, arity=2)
    print([tokenizer.id_to_token[id] for id in valid])

    print("\nValid replacements for TERMINAL:")
    valid = tokenizer.get_valid_replacements(NodeType.TERMINAL)
    print([tokenizer.id_to_token[id] for id in valid])
