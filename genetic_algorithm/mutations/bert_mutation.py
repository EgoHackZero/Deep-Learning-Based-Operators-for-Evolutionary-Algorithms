"""
BERT Mutation Operator

Implements a learned mutation operator using a BERT-style transformer model.
The model is trained with REINFORCE to predict better replacements for masked nodes.
"""

import torch
import torch.nn.functional as F
from typing import List, Optional, Tuple, Any
import random
import numpy as np

from genetic_algorithm.mutations.base_mutation import BaseMutation
from genetic_algorithm.bert_model import BERTModel, BERTMutationPolicy
from genetic_algorithm.tokenizer import BaseTokenizer
from genetic_algorithm.gp_tree import GPTree, NodeType


class BERTMutation(BaseMutation):
    """
    BERT-based mutation operator for GP trees.

    Process:
    1. Convert individual to GPTree
    2. Randomly mask nodes with masking_prob
    3. Replace masks sequentially in DFS order
    4. For each mask:
        - Use BERT to predict distribution over valid replacements
        - Sample a replacement token
        - Update the sequence
    5. Convert back to individual format
    """

    def __init__(
        self,
        tokenizer: BaseTokenizer,
        masking_prob: float = 0.15,
        temperature: float = 1.0,
        epsilon_greedy: float = 0.1,
        embedding_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        device: Optional[torch.device] = None,
        chance: float = 1.0,  # Different default from BaseMutation
        seed: Optional[int] = None
    ):
        """
        Initialize BERT mutation operator.

        Args:
            tokenizer: Tokenizer for the problem domain
            masking_prob: Probability of masking each node
            temperature: Temperature for sampling (higher = more random)
            epsilon_greedy: Probability of random exploration (0.0-1.0, default 0.1)
            embedding_dim: BERT embedding dimension
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            dim_feedforward: Feedforward dimension
            dropout: Dropout probability
            device: Device to run on (cuda/cpu)
            chance: Probability of applying mutation
            seed: Random seed
        """
        super().__init__(chance=chance, seed=seed)

        self.tokenizer = tokenizer
        self.masking_prob = masking_prob
        self.temperature = temperature
        self.epsilon_greedy = epsilon_greedy
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize BERT model
        self.model = BERTModel(
            vocab_size=tokenizer.vocab_size,
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        ).to(self.device)

        # Policy wrapper
        self.policy = BERTMutationPolicy(self.model, self.device)

        # Training mode flag
        self._training_mode = False

        # Cache for training
        self.last_mutation_info = None

    def train_mode(self):
        """Set to training mode"""
        self._training_mode = True
        self.model.train()

    def eval_mode(self):
        """Set to evaluation mode"""
        self._training_mode = False
        self.model.eval()

    def _mutate(self, offspring: list, **kwargs) -> list:
        """
        Perform BERT mutation on an individual.

        Args:
            offspring: Individual to mutate (flat list)
            **kwargs: Additional parameters

        Returns:
            Mutated individual (flat list)
        """
        try:
            # Convert to GPTree
            tree = self._list_to_gptree(offspring)

            # Apply BERT mutation
            mutated_tree, mutation_info = self._mutate_tree(tree)

            # Store mutation info for training
            if self._training_mode:
                self.last_mutation_info = mutation_info

            # Convert back to list
            return mutated_tree.to_list()
        except (ValueError, Exception) as e:
            # If tree is invalid or mutation fails, return offspring unchanged
            # This can happen when DNC crossover produces invalid trees
            if self._training_mode:
                self.last_mutation_info = None
            return offspring

    def _list_to_gptree(self, individual: list) -> GPTree:
        """
        Convert flat list individual to GPTree with metadata.

        For now, assumes the individual is already in prefix notation.
        Subclasses can override this for problem-specific conversion.

        Args:
            individual: Flat list representation

        Returns:
            GPTree with metadata
        """
        # Build metadata based on token types
        metadata = []
        for value in individual:
            if value in self.tokenizer.get_function_arities():
                # Function node
                metadata.append({
                    'type': NodeType.FUNCTION,
                    'arity': self.tokenizer.get_function_arities()[value],
                    'value': value
                })
            elif value in self.tokenizer.get_terminal_set():
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

        return GPTree(individual.copy(), metadata)

    def _mutate_tree(self, tree: GPTree) -> Tuple[GPTree, dict]:
        """
        Apply BERT mutation to a GP tree.

        Args:
            tree: GPTree to mutate

        Returns:
            Tuple of (mutated_tree, mutation_info)
            mutation_info contains data needed for training
        """
        # Encode tree to tokens
        token_ids, metadata = self.tokenizer.encode(tree)
        original_token_ids = token_ids.clone()

        # Select positions to mask (DFS order)
        dfs_order = tree.get_dfs_order()
        mask_positions = self._select_mask_positions(dfs_order)

        if not mask_positions:
            # No positions to mask, return original
            return tree, None

        # Store info for training
        mutation_info = {
            'original_token_ids': original_token_ids,
            'mask_positions': mask_positions,
            'sampled_tokens': [],
            'valid_tokens_list': [],
            'metadata': metadata
        }

        # Replace masks sequentially in DFS order
        current_tokens = token_ids.clone()

        for mask_pos in mask_positions:
            # Mask this position
            current_tokens[mask_pos] = self.tokenizer.mask_token_id

            # Get valid replacements based on node type
            node_meta = metadata[mask_pos]
            node_type = node_meta['type']
            arity = node_meta.get('arity', 0) if node_type == NodeType.FUNCTION else None

            valid_token_ids = self.tokenizer.get_valid_replacements(node_type, arity)

            if not valid_token_ids:
                # No valid replacements, skip
                continue

            # Predict replacement using BERT
            with torch.no_grad() if not self._training_mode else torch.enable_grad():
                current_tokens_batch = current_tokens.unsqueeze(0).to(self.device)
                attention_mask = self.tokenizer.create_attention_mask(current_tokens_batch)

                sampled_id, log_prob, probs = self.model.predict_masked_token(
                    current_tokens_batch,
                    mask_position=mask_pos,
                    attention_mask=attention_mask,
                    temperature=self.temperature,
                    valid_token_ids=valid_token_ids,
                    epsilon_greedy=self.epsilon_greedy if self._training_mode else 0.0
                )

            # Replace mask with sampled token
            current_tokens[mask_pos] = sampled_id

            # Store for training
            mutation_info['sampled_tokens'].append(sampled_id)
            mutation_info['valid_tokens_list'].append(valid_token_ids)

        # Decode tokens back to tree
        mutated_tree = self.tokenizer.decode(current_tokens, metadata)

        return mutated_tree, mutation_info

    def _select_mask_positions(self, dfs_order: List[int]) -> List[int]:
        """
        Select positions to mask based on masking probability.

        Args:
            dfs_order: Indices in DFS order

        Returns:
            List of positions to mask (in DFS order)
        """
        mask_positions = []

        for pos in dfs_order:
            if random.random() < self.masking_prob:
                mask_positions.append(pos)

        return mask_positions

    def get_parameters(self):
        """Get model parameters for optimizer"""
        return self.model.parameters()

    def save_model(self, path: str):
        """
        Save model weights.

        Args:
            path: Path to save model
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'vocab_size': self.tokenizer.vocab_size,
            'masking_prob': self.masking_prob,
            'temperature': self.temperature,
            'epsilon_greedy': self.epsilon_greedy
        }, path)
        print(f"BERT mutation model saved to {path}")

    def load_model(self, path: str):
        """
        Load model weights.

        Args:
            path: Path to load model from
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        # Handle both formats: 'model_state_dict' (from save_model) and 'model' (from trainer)
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        elif 'model' in checkpoint:
            self.model.load_state_dict(checkpoint['model'])
        else:
            raise KeyError("Checkpoint must contain either 'model_state_dict' or 'model' key")

        self.masking_prob = checkpoint.get('masking_prob', self.masking_prob)
        self.temperature = checkpoint.get('temperature', self.temperature)
        self.epsilon_greedy = checkpoint.get('epsilon_greedy', self.epsilon_greedy)
        print(f"BERT mutation model loaded from {path}")

    def compute_log_probs_batch(
        self,
        individuals: List[list],
        mutation_infos: List[dict]
    ) -> torch.Tensor:
        """
        Compute log probabilities for a batch of mutations.
        Used during training for policy gradient.

        Args:
            individuals: List of individuals (flat lists)
            mutation_infos: List of mutation info dicts

        Returns:
            Tensor of log probabilities [batch_size]
        """
        # Prepare batch data
        token_ids_batch = []
        mask_positions_batch = []
        sampled_tokens_batch = []
        valid_tokens_batch = []

        for info in mutation_infos:
            if info is None:
                continue

            token_ids_batch.append(info['original_token_ids'])
            mask_positions_batch.append(info['mask_positions'])
            sampled_tokens_batch.append(info['sampled_tokens'])
            valid_tokens_batch.append(info['valid_tokens_list'])

        # Compute log probs
        log_probs = self.policy.compute_action_log_probs(
            token_ids_batch,
            mask_positions_batch,
            sampled_tokens_batch,
            valid_tokens_batch
        )

        return log_probs


class SymbolicRegressionBERTMutation(BERTMutation):
    """
    BERT mutation specialized for symbolic regression problems.

    Handles constant generation when CONST_TOKEN is sampled.
    """

    def __init__(
        self,
        tokenizer: BaseTokenizer,
        constant_range: Tuple[float, float] = (-10.0, 10.0),
        **kwargs
    ):
        """
        Initialize symbolic regression BERT mutation.

        Args:
            tokenizer: Tokenizer for symbolic regression
            constant_range: Range for generating constants
            **kwargs: Additional arguments for BERTMutation
        """
        super().__init__(tokenizer, **kwargs)
        self.constant_range = constant_range

    def _mutate_tree(self, tree: GPTree) -> Tuple[GPTree, dict]:
        """
        Apply mutation with constant handling.

        Args:
            tree: GPTree to mutate

        Returns:
            Tuple of (mutated_tree, mutation_info)
        """
        # Call parent mutation
        mutated_tree, mutation_info = super()._mutate_tree(tree)

        # Handle constants: replace CONST_TOKEN with actual values
        for i, value in enumerate(mutated_tree.prefix_sequence):
            if value is None or (isinstance(value, str) and value == "CONST_TOKEN"):
                # Generate random constant
                mutated_tree.prefix_sequence[i] = np.random.uniform(*self.constant_range)

        return mutated_tree, mutation_info


if __name__ == "__main__":
    # Test BERT mutation
    print("Testing BERT Mutation...")

    from genetic_algorithm.tokenizer import SymbolicRegressionTokenizer
    from genetic_algorithm.gp_tree import generate_random_tree

    # Create tokenizer
    tokenizer = SymbolicRegressionTokenizer(terminal_names=['x', 'y'])

    # Create mutation operator
    mutation = SymbolicRegressionBERTMutation(
        tokenizer=tokenizer,
        masking_prob=0.3,
        embedding_dim=32,
        num_heads=2,
        num_layers=1,
        constant_range=(-5.0, 5.0)
    )

    # Generate a random tree
    tree = generate_random_tree(
        max_depth=3,
        function_set=tokenizer.get_function_arities(),
        terminal_set=list(tokenizer.get_terminal_set()),
        constant_range=(-5.0, 5.0)
    )

    print(f"\nOriginal tree: {tree}")
    print(f"Length: {len(tree)}")

    # Apply mutation
    individual = tree.to_list()
    mutated = mutation.perform(individual)

    print(f"\nMutated individual: {mutated}")
    print(f"Length: {len(mutated)}")

    # Test with training mode
    print("\nTesting training mode...")
    mutation.train_mode()
    mutated2 = mutation.perform(individual)
    print(f"Mutation info stored: {mutation.last_mutation_info is not None}")

    print("\nBERT Mutation test passed!")
