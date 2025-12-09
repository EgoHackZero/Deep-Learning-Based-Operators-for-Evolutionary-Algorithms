"""
BERT Model for GP Tree Mutation

Implements a Transformer-based model for predicting masked nodes in GP trees.
Uses REINFORCE for training since there's no ground-truth labels.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
import math


class PositionalEncoding(nn.Module):
    """
    Adds positional information to token embeddings.
    Uses sinusoidal encoding as in "Attention is All You Need".
    Works with batch_first format [batch_size, seq_len, embedding_dim].
    """

    def __init__(self, embedding_dim: int, max_len: int = 500, dropout: float = 0.1):
        """
        Initialize positional encoding.

        Args:
            embedding_dim: Dimension of embeddings
            max_len: Maximum sequence length
            dropout: Dropout probability
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix [1, max_len, embedding_dim]
        position = torch.arange(max_len).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(torch.arange(0, embedding_dim, 2) * (-math.log(10000.0) / embedding_dim))

        pe = torch.zeros(1, max_len, embedding_dim)  # [1, max_len, embedding_dim]
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.

        Args:
            x: Input tensor [batch_size, seq_len, embedding_dim]

        Returns:
            Tensor with positional encoding added [batch_size, seq_len, embedding_dim]
        """
        # x: [batch_size, seq_len, embedding_dim]
        # self.pe: [1, max_len, embedding_dim]
        x = x + self.pe[:, :x.size(1), :]  # Broadcast across batch and select up to seq_len
        return self.dropout(x)


class BERTModel(nn.Module):
    """
    BERT-style Transformer model for GP tree mutation.

    Architecture:
        Input: Token IDs [batch_size, seq_len]
        ↓
        Token Embedding [batch_size, seq_len, embedding_dim]
        ↓
        Positional Encoding
        ↓
        Transformer Encoder (N layers)
        ↓
        MLM Head (Linear projection to vocab_size)
        ↓
        Output: Logits [batch_size, seq_len, vocab_size]
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        max_seq_len: int = 100
    ):
        """
        Initialize BERT model.

        Args:
            vocab_size: Size of the vocabulary
            embedding_dim: Dimension of token embeddings
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            dim_feedforward: Dimension of feedforward network
            dropout: Dropout probability
            max_seq_len: Maximum sequence length
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(embedding_dim, max_seq_len, dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True  # Input shape: [batch, seq, feature]
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Masked language modeling head
        self.mlm_head = nn.Linear(embedding_dim, vocab_size)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights"""
        # Initialize embeddings
        nn.init.normal_(self.token_embedding.weight, mean=0, std=0.02)

        # Initialize MLM head
        nn.init.normal_(self.mlm_head.weight, mean=0, std=0.02)
        if self.mlm_head.bias is not None:
            nn.init.zeros_(self.mlm_head.bias)

    def forward(
        self,
        token_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            token_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
                           1 for real tokens, 0 for padding

        Returns:
            Logits [batch_size, seq_len, vocab_size]
        """
        # Token embeddings [batch_size, seq_len, embedding_dim]
        embeddings = self.token_embedding(token_ids)

        # Add positional encoding (works with batch_first format)
        embeddings = self.pos_encoder(embeddings)  # [batch_size, seq_len, embedding_dim]

        # Create padding mask for transformer
        # TransformerEncoder expects: True for positions to IGNORE
        if attention_mask is not None:
            # Convert: 1 (attend) -> False, 0 (ignore) -> True
            src_key_padding_mask = (attention_mask == 0)
        else:
            src_key_padding_mask = None

        # Pass through transformer encoder
        encoder_output = self.transformer_encoder(
            embeddings,
            src_key_padding_mask=src_key_padding_mask
        )

        # Project to vocabulary
        logits = self.mlm_head(encoder_output)

        return logits

    def predict_masked_token(
        self,
        token_ids: torch.Tensor,
        mask_position: int,
        attention_mask: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
        valid_token_ids: Optional[List[int]] = None,
        epsilon_greedy: float = 0.0
    ) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """
        Predict a replacement for a masked token.

        Args:
            token_ids: Token IDs with one [MASK] token [batch_size, seq_len]
            mask_position: Position of the mask in the sequence
            attention_mask: Attention mask [batch_size, seq_len]
            temperature: Temperature for sampling (higher = more random)
            valid_token_ids: List of valid token IDs to consider
            epsilon_greedy: Probability of random exploration (0.0 = no exploration)

        Returns:
            Tuple of (sampled_token_id, log_prob, distribution)
        """
        # Forward pass
        logits = self.forward(token_ids, attention_mask)

        # Get logits for the masked position [batch_size, vocab_size]
        mask_logits = logits[:, mask_position, :]

        # Apply temperature
        mask_logits = mask_logits / temperature

        # Apply constraint mask if provided
        if valid_token_ids is not None:
            # Create mask: -inf for invalid tokens, 0 for valid tokens
            constraint_mask = torch.full_like(mask_logits, float('-inf'))
            constraint_mask[:, valid_token_ids] = 0
            mask_logits = mask_logits + constraint_mask

        # Compute probabilities
        probs = F.softmax(mask_logits, dim=-1)

        # Epsilon-greedy exploration
        if epsilon_greedy > 0 and torch.rand(1).item() < epsilon_greedy:
            # Explore: uniformly sample from valid tokens
            if valid_token_ids is not None and len(valid_token_ids) > 0:
                sampled_token_id = torch.tensor(valid_token_ids[torch.randint(0, len(valid_token_ids), (1,)).item()])
            else:
                # Sample from all tokens
                sampled_token_id = torch.randint(0, self.vocab_size, (1,))

            # Compute log probability of the sampled token
            log_prob = torch.log(probs[0, sampled_token_id] + 1e-10)
        else:
            # Exploit: sample from distribution
            dist = torch.distributions.Categorical(probs)
            sampled_token_id = dist.sample()
            log_prob = dist.log_prob(sampled_token_id)

        return sampled_token_id.item(), log_prob, probs

    def get_distribution_for_mask(
        self,
        token_ids: torch.Tensor,
        mask_position: int,
        attention_mask: Optional[torch.Tensor] = None,
        valid_token_ids: Optional[List[int]] = None
    ) -> torch.Tensor:
        """
        Get probability distribution for a masked position.

        Args:
            token_ids: Token IDs with [MASK] token [batch_size, seq_len]
            mask_position: Position of the mask
            attention_mask: Attention mask
            valid_token_ids: List of valid token IDs

        Returns:
            Probability distribution [batch_size, vocab_size]
        """
        # Forward pass
        logits = self.forward(token_ids, attention_mask)

        # Get logits for masked position
        mask_logits = logits[:, mask_position, :]

        # Apply constraints
        if valid_token_ids is not None:
            constraint_mask = torch.full_like(mask_logits, float('-inf'))
            constraint_mask[:, valid_token_ids] = 0
            mask_logits = mask_logits + constraint_mask

        # Return probabilities
        return F.softmax(mask_logits, dim=-1)


class BERTMutationPolicy:
    """
    Wrapper class that handles the mutation policy logic.
    Manages sequential mask replacement with DFS ordering.
    """

    def __init__(self, model: BERTModel, device: torch.device):
        """
        Initialize policy.

        Args:
            model: BERT model
            device: Device to run on
        """
        self.model = model
        self.device = device

    def compute_action_log_probs(
        self,
        token_ids_batch: List[torch.Tensor],
        mask_positions_batch: List[List[int]],
        sampled_tokens_batch: List[List[int]],
        valid_tokens_batch: List[List[List[int]]]
    ) -> torch.Tensor:
        """
        Compute log probabilities of sampled actions for a batch.

        This is used during training to compute the policy gradient.

        Args:
            token_ids_batch: List of token ID tensors (can have different lengths)
            mask_positions_batch: List of mask position lists for each sequence
            sampled_tokens_batch: List of sampled token lists for each sequence
            valid_tokens_batch: List of valid token lists for each mask

        Returns:
            Tensor of log probabilities for each sequence in batch
        """
        batch_log_probs = []

        for token_ids, mask_positions, sampled_tokens, valid_tokens_list in zip(
            token_ids_batch, mask_positions_batch, sampled_tokens_batch, valid_tokens_batch
        ):
            sequence_log_probs = []

            # Current token sequence
            current_tokens = token_ids.clone()

            # Replace masks sequentially
            for mask_pos, sampled_token, valid_tokens in zip(
                mask_positions, sampled_tokens, valid_tokens_list
            ):
                # Get distribution for this mask
                current_tokens = current_tokens.unsqueeze(0).to(self.device)  # Add batch dim
                logits = self.model(current_tokens)

                # Get logits for masked position
                mask_logits = logits[0, mask_pos, :]  # [vocab_size]

                # Apply constraints
                if valid_tokens:
                    constraint_mask = torch.full_like(mask_logits, float('-inf'))
                    constraint_mask[valid_tokens] = 0
                    mask_logits = mask_logits + constraint_mask

                # Compute log probability of sampled token
                log_probs = F.log_softmax(mask_logits, dim=-1)
                sequence_log_probs.append(log_probs[sampled_token])

                # Replace mask with sampled token
                current_tokens[0, mask_pos] = sampled_token

            # Sum log probs for this sequence
            batch_log_probs.append(torch.stack(sequence_log_probs).sum())

        return torch.stack(batch_log_probs)


if __name__ == "__main__":
    # Test the BERT model
    print("Testing BERT Model...")

    vocab_size = 20
    batch_size = 2
    seq_len = 10

    model = BERTModel(
        vocab_size=vocab_size,
        embedding_dim=32,
        num_heads=4,
        num_layers=2
    )

    # Create dummy input
    token_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)

    print(f"Input shape: {token_ids.shape}")

    # Forward pass
    logits = model(token_ids, attention_mask)
    print(f"Output shape: {logits.shape}")
    print(f"Expected shape: [{batch_size}, {seq_len}, {vocab_size}]")

    # Test prediction
    mask_token_id = 1
    token_ids[0, 5] = mask_token_id
    valid_tokens = [3, 4, 5, 6]

    sampled_id, log_prob, probs = model.predict_masked_token(
        token_ids,
        mask_position=5,
        attention_mask=attention_mask,
        valid_token_ids=valid_tokens
    )

    print(f"\nPrediction test:")
    print(f"Sampled token ID: {sampled_id}")
    print(f"Log probability: {log_prob}")
    print(f"Valid tokens had non-zero prob: {probs[0, valid_tokens].sum() > 0}")

    print("\nBERT Model test passed!")
