"""
RL Trainer for BERT Mutation Operator

Implements REINFORCE training for the BERT mutation operator.
"""

import torch
import torch.optim as optim
from typing import Callable, Optional, List
import numpy as np
from tqdm import tqdm
import os


class BERTMutationTrainer:
    """
    Reinforcement learning trainer for BERT mutation operator.

    Uses REINFORCE (policy gradient) to train the BERT model to generate
    mutations that improve fitness.
    """

    def __init__(
        self,
        mutation_operator,  # BERTMutation instance
        fitness_function: Callable,
        learning_rate: float = 1e-4,
        device: Optional[torch.device] = None,
        baseline_type: str = "moving_average",
        baseline_alpha: float = 0.1,
        reward_clip: Optional[float] = None
    ):
        """
        Initialize BERT mutation trainer.

        Args:
            mutation_operator: BERTMutation instance to train
            fitness_function: Function to evaluate fitness of individuals
            learning_rate: Learning rate for optimizer
            device: Device to train on (cuda/cpu)
            baseline_type: Type of baseline ('moving_average', 'batch_mean', or None)
            baseline_alpha: Alpha for moving average baseline
            reward_clip: Optional value to clip rewards (helps stability)
        """
        self.mutation = mutation_operator
        self.fitness_function = fitness_function
        self.baseline_type = baseline_type
        self.baseline_alpha = baseline_alpha
        self.reward_clip = reward_clip
        self.baseline = 0.0

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        # Optimizer for BERT model
        self.optimizer = optim.Adam(
            self.mutation.model.parameters(),
            lr=learning_rate
        )

        # Training history
        self.history = {
            'rewards': [],
            'losses': [],
            'avg_fitness': [],
            'best_fitness': [],
            'baseline': []
        }

    def _compute_reward(
        self,
        mutated_individual: list,
        original_individual: list,
        reward_type: str = "improvement"
    ) -> float:
        """
        Compute reward based on fitness improvement.

        Args:
            mutated_individual: Individual after mutation
            original_individual: Individual before mutation
            reward_type: Type of reward computation
                - "improvement": reward = fitness(mutated) - fitness(original)
                - "absolute": reward = fitness(mutated)
                - "relative": reward = fitness(mutated) / fitness(original)

        Returns:
            Reward value
        """
        mutated_fitness = self.fitness_function(mutated_individual)

        if reward_type == "absolute":
            return mutated_fitness

        original_fitness = self.fitness_function(original_individual)

        if reward_type == "improvement":
            return mutated_fitness - original_fitness
        elif reward_type == "relative":
            if original_fitness == 0:
                return mutated_fitness
            return mutated_fitness / original_fitness
        else:
            raise ValueError(f"Unknown reward type: {reward_type}")

    def _update_baseline(self, reward: float):
        """Update baseline using moving average"""
        if self.baseline_type == "moving_average":
            self.baseline = (1 - self.baseline_alpha) * self.baseline + self.baseline_alpha * reward

    def train_episode(
        self,
        individuals: List[list],
        temperature: float = 1.0,
        reward_type: str = "improvement"
    ) -> dict:
        """
        Train on a batch of individuals.

        Args:
            individuals: List of individuals to mutate
            temperature: Temperature for sampling
            reward_type: Type of reward computation

        Returns:
            Dictionary with episode statistics
        """
        self.mutation.train_mode()
        self.mutation.temperature = temperature

        batch_rewards = []
        batch_log_probs = []
        fitness_values = []
        mutation_infos = []

        # Collect episodes (mutations)
        for individual in individuals:
            # Store original
            original = individual.copy()

            # Apply mutation and collect info
            mutated = self.mutation.perform(individual.copy())
            mutation_info = self.mutation.last_mutation_info

            # Skip if no mutation occurred
            if mutation_info is None or not mutation_info['mask_positions']:
                continue

            # Compute reward
            reward = self._compute_reward(mutated, original, reward_type)

            # Clip reward if specified
            if self.reward_clip is not None:
                reward = np.clip(reward, -self.reward_clip, self.reward_clip)

            batch_rewards.append(reward)
            mutation_infos.append(mutation_info)

            # Track fitness
            mutated_fitness = self.fitness_function(mutated)
            fitness_values.append(mutated_fitness)

        # Check if we have any valid mutations
        if not batch_rewards:
            return {
                'loss': 0.0,
                'avg_reward': 0.0,
                'avg_fitness': 0.0,
                'best_fitness': 0.0,
                'baseline': self.baseline
            }

        # Compute baseline
        if self.baseline_type == "batch_mean":
            baseline = np.mean(batch_rewards)
        elif self.baseline_type == "moving_average":
            baseline = self.baseline
        else:
            baseline = 0.0

        # Convert batch_rewards to numpy array for easier manipulation
        batch_rewards_np = np.array(batch_rewards)

        # Normalize advantages (mean + std normalization as in reference implementation)
        # This reduces variance and improves training stability
        rewards_std = np.std(batch_rewards_np)
        if rewards_std > 1e-10:  # Avoid division by zero
            advantages = (batch_rewards_np - baseline) / rewards_std
        else:
            advantages = batch_rewards_np - baseline

        # Compute policy gradient loss
        batch_loss = 0.0

        for i, (individual, mutation_info) in enumerate(zip(individuals[:len(batch_rewards)], mutation_infos)):
            if mutation_info is None:
                continue

            # Compute log probabilities for this mutation
            log_probs = self._compute_log_probs_for_mutation(mutation_info)

            # REINFORCE loss: -log_prob * advantage
            loss = -torch.sum(log_probs) * advantages[i]
            batch_loss += loss

        # Average loss over batch
        if len(batch_rewards) > 0:
            batch_loss = batch_loss / len(batch_rewards)
        else:
            batch_loss = torch.tensor(0.0)

        # Backpropagation
        self.optimizer.zero_grad()
        if batch_loss.requires_grad:
            batch_loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.mutation.model.parameters(),
                max_norm=1.0
            )

            self.optimizer.step()

        # Update baseline
        if self.baseline_type == "moving_average":
            self._update_baseline(np.mean(batch_rewards))

        self.mutation.eval_mode()

        return {
            'loss': batch_loss.item() if isinstance(batch_loss, torch.Tensor) else batch_loss,
            'avg_reward': np.mean(batch_rewards) if batch_rewards else 0.0,
            'avg_fitness': np.mean(fitness_values) if fitness_values else 0.0,
            'best_fitness': np.max(fitness_values) if fitness_values else 0.0,
            'baseline': baseline,
            'num_mutations': len(batch_rewards)
        }

    def _compute_log_probs_for_mutation(self, mutation_info: dict) -> torch.Tensor:
        """
        Compute log probabilities for a mutation.

        Args:
            mutation_info: Dictionary with mutation information

        Returns:
            Tensor of log probabilities
        """
        original_tokens = mutation_info['original_token_ids']
        mask_positions = mutation_info['mask_positions']
        sampled_tokens = mutation_info['sampled_tokens']
        valid_tokens_list = mutation_info['valid_tokens_list']
        metadata = mutation_info['metadata']

        log_probs_list = []
        current_tokens = original_tokens.clone()

        # Replace masks sequentially and compute log probs
        for mask_pos, sampled_token, valid_tokens in zip(
            mask_positions, sampled_tokens, valid_tokens_list
        ):
            # Create a new tensor with mask at this position (avoid in-place op)
            masked_tokens = current_tokens.clone()
            masked_tokens[mask_pos] = self.mutation.tokenizer.mask_token_id

            # Get model output
            masked_tokens_batch = masked_tokens.unsqueeze(0).to(self.device)
            logits = self.mutation.model(masked_tokens_batch)

            # Get logits for masked position
            mask_logits = logits[0, mask_pos, :]

            # Apply constraints
            if valid_tokens:
                constraint_mask = torch.full_like(mask_logits, float('-inf'))
                constraint_mask[valid_tokens] = 0
                mask_logits = mask_logits + constraint_mask

            # Compute log probability of sampled token
            log_probs = torch.nn.functional.log_softmax(mask_logits, dim=-1)
            log_probs_list.append(log_probs[sampled_token])

            # Create new tensor with sampled token (avoid in-place op)
            current_tokens = current_tokens.clone()
            current_tokens[mask_pos] = sampled_token

        return torch.stack(log_probs_list)

    def train(
        self,
        population_generator: Callable,
        num_episodes: int,
        batch_size: int = 32,
        temperature: float = 1.0,
        reward_type: str = "improvement",
        save_interval: int = 100,
        save_path: str = "models/bert_mutation_checkpoint.pth",
        verbose: bool = True
    ):
        """
        Main training loop.

        Args:
            population_generator: Function that generates list of individuals
            num_episodes: Number of training episodes
            batch_size: Number of individuals per episode
            temperature: Temperature for sampling
            reward_type: Type of reward computation
            save_interval: Save model every N episodes
            save_path: Path to save model checkpoints
            verbose: Whether to print training progress
        """
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)

        if verbose:
            pbar = tqdm(range(num_episodes), desc="Training BERT Mutation")
        else:
            pbar = range(num_episodes)

        for episode in pbar:
            # Generate individuals
            individuals = population_generator(batch_size)

            # Train on batch
            stats = self.train_episode(
                individuals,
                temperature=temperature,
                reward_type=reward_type
            )

            # Update history
            self.history['losses'].append(stats['loss'])
            self.history['rewards'].append(stats['avg_reward'])
            self.history['avg_fitness'].append(stats['avg_fitness'])
            self.history['best_fitness'].append(stats['best_fitness'])
            self.history['baseline'].append(stats['baseline'])

            # Update progress bar
            if verbose:
                pbar.set_postfix({
                    'loss': f"{stats['loss']:.4f}",
                    'reward': f"{stats['avg_reward']:.4f}",
                    'fitness': f"{stats['avg_fitness']:.4f}",
                    'mutations': stats['num_mutations']
                })

            # Save checkpoint
            if (episode + 1) % save_interval == 0:
                checkpoint_path = save_path.replace('.pth', f'_ep{episode+1}.pth')
                self.save_checkpoint(checkpoint_path, episode)
                if verbose:
                    print(f"\nCheckpoint saved to {checkpoint_path}")

        # Save final model
        self.save_checkpoint(save_path, num_episodes)
        if verbose:
            print(f"\nFinal model saved to {save_path}")

    def save_checkpoint(self, path: str, episode: int):
        """Save training checkpoint"""
        torch.save({
            'episode': episode,
            'model': self.mutation.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'history': self.history,
            'baseline': self.baseline,
            'vocab_size': self.mutation.tokenizer.vocab_size,
            'masking_prob': self.mutation.masking_prob
        }, path)

    def load_checkpoint(self, path: str):
        """Load training checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.mutation.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.history = checkpoint['history']
        self.baseline = checkpoint['baseline']
        return checkpoint['episode']


if __name__ == "__main__":
    # Test the trainer
    print("Testing BERT Mutation Trainer...")

    from genetic_algorithm.tokenizer import SymbolicRegressionTokenizer
    from genetic_algorithm.mutations.bert_mutation import SymbolicRegressionBERTMutation
    from genetic_algorithm.gp_tree import generate_random_tree

    # Create tokenizer
    tokenizer = SymbolicRegressionTokenizer(terminal_names=['x'])

    # Create mutation operator
    mutation = SymbolicRegressionBERTMutation(
        tokenizer=tokenizer,
        masking_prob=0.2,
        embedding_dim=32,
        num_heads=2,
        num_layers=1
    )

    # Simple fitness function (tree size - smaller is better, so negate)
    def fitness_fn(individual):
        return -len(individual)  # Reward shorter trees

    # Create trainer
    trainer = BERTMutationTrainer(
        mutation_operator=mutation,
        fitness_function=fitness_fn,
        learning_rate=1e-3
    )

    # Population generator
    def gen_population(size):
        individuals = []
        for _ in range(size):
            tree = generate_random_tree(
                max_depth=3,
                function_set=tokenizer.get_function_arities(),
                terminal_set=list(tokenizer.get_terminal_set()),
                constant_range=(-5.0, 5.0)
            )
            individuals.append(tree.to_list())
        return individuals

    # Train for a few episodes
    print("\nTraining for 5 episodes...")
    trainer.train(
        population_generator=gen_population,
        num_episodes=5,
        batch_size=8,
        verbose=True
    )

    print("\nBERT Mutation Trainer test passed!")
