import torch
import torch.optim as optim
from typing import Callable, Optional, List, Tuple
import numpy as np
from tqdm import tqdm
import os


class RLTrainer:
    def __init__(
        self,
        operator,
        fitness_function: Callable,
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        device: Optional[torch.device] = None,
        baseline_type: str = "moving_average",
        baseline_alpha: float = 0.1
    ):
        """
        Reinforcement Learning trainer for DL-based genetic operators.
        
        Args:
            operator: DNCrossover instance to train
            fitness_function: Fitness function to evaluate offspring quality
            learning_rate: Learning rate for optimizer
            gamma: Discount factor (not used for single-step episodes, kept for completeness)
            device: Device to train on
            baseline_type: Type of baseline ('moving_average', 'batch_mean', or None)
            baseline_alpha: Alpha for moving average baseline
        """
        self.operator = operator
        self.fitness_function = fitness_function
        self.gamma = gamma
        self.baseline_type = baseline_type
        self.baseline_alpha = baseline_alpha
        self.baseline = 0.0
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        # optimizer for both encoder and decoder
        self.optimizer = optim.Adam(
            list(self.operator.encoder.parameters()) + 
            list(self.operator.decoder.parameters()),
            lr=learning_rate
        )
        
        # training history
        self.history = {
            'rewards': [],
            'losses': [],
            'avg_fitness': [],
            'best_fitness': []
        }
    
    def _compute_reward(
        self, 
        offspring: list, 
        parent1: list, 
        parent2: list,
        reward_type: str = "improvement"
    ) -> float:
        """
        Computes reward based on offspring fitness.
        
        Args:
            offspring: Generated offspring
            parent1: First parent
            parent2: Second parent
            reward_type: Type of reward computation
                - "improvement": reward = fitness(offspring) - avg(fitness(parents))
                - "absolute": reward = fitness(offspring)
                - "relative": reward = fitness(offspring) / avg(fitness(parents))
        
        Returns:
            Reward value
        """
        offspring_fitness = self.fitness_function(offspring)
        
        if reward_type == "absolute":
            return offspring_fitness
        
        parent1_fitness = self.fitness_function(parent1)
        parent2_fitness = self.fitness_function(parent2)
        avg_parent_fitness = (parent1_fitness + parent2_fitness) / 2
        
        if reward_type == "improvement":
            return offspring_fitness - avg_parent_fitness
        elif reward_type == "relative":
            if avg_parent_fitness == 0:
                return offspring_fitness
            return offspring_fitness / avg_parent_fitness
        else:
            raise ValueError(f"Unknown reward type: {reward_type}")
    
    def _forward_with_log_probs(
        self, 
        parent1: list, 
        parent2: list,
        temperature: float = 1.0
    ) -> Tuple[list, torch.Tensor, torch.Tensor]:
        """
        Forward pass that returns offspring, log probabilities, and action probabilities.
        
        Args:
            parent1: First parent
            parent2: Second parent
            temperature: Temperature for sampling
        
        Returns:
            offspring: Generated offspring
            log_probs: Log probabilities of selected actions
            action_probs: Action probability distribution
        """
        # prepare inputs
        p1_tensor = self.operator._prepare_input(parent1)
        p2_tensor = self.operator._prepare_input(parent2)
        
        # encode both parents
        p1_encoded, p1_hidden = self.operator.encoder(p1_tensor)
        p2_encoded, p2_hidden = self.operator.encoder(p2_tensor)
        
        # combine parent encodings
        combined_encoded = (p1_encoded + p2_encoded) / 2
        h_combined = (p1_hidden[0] + p2_hidden[0]) / 2
        c_combined = (p1_hidden[1] + p2_hidden[1]) / 2
        combined_hidden = (h_combined, c_combined)
        
        # decode to get gene selection logits
        logits = self.operator.decoder(combined_encoded, combined_hidden)
        
        # apply temperature scaling and softmax
        probs = torch.softmax(logits / temperature, dim=-1)  # shape: (1, seq_len, 2)
        
        # sample actions and compute log probabilities
        offspring = []
        log_probs_list = []
        
        for i in range(len(parent1)):
            # get probability distribution for this position
            prob_dist = probs[0, i]  # shape: (2,)
            
            # sample action (0=parent1, 1=parent2)
            action_dist = torch.distributions.Categorical(prob_dist)
            action = action_dist.sample()
            
            # get log probability of sampled action
            log_prob = action_dist.log_prob(action)
            log_probs_list.append(log_prob)
            
            # select gene based on action
            if action.item() == 0:
                offspring.append(parent1[i])
            else:
                offspring.append(parent2[i])
        
        # stack log probabilities
        log_probs = torch.stack(log_probs_list)
        
        return offspring, log_probs, probs
    
    def _update_baseline(self, reward: float):
        """Updates the baseline using moving average"""
        if self.baseline_type == "moving_average":
            self.baseline = (1 - self.baseline_alpha) * self.baseline + self.baseline_alpha * reward
    
    def train_episode(
        self,
        parent_pairs: List[Tuple[list, list]],
        temperature: float = 1.0,
        reward_type: str = "improvement"
    ) -> dict:
        """
        Trains on a batch of parent pairs.
        
        Args:
            parent_pairs: List of (parent1, parent2) tuples
            temperature: Temperature for sampling
            reward_type: Type of reward computation
        
        Returns:
            Dictionary with episode statistics
        """
        self.operator.train_mode()
        
        total_reward = 0.0
        batch_rewards = []
        fitness_values = []
        
        # collect episodes
        for parent1, parent2 in parent_pairs:
            # forward pass with log probabilities
            offspring, log_probs, _ = self._forward_with_log_probs(
                parent1, parent2, temperature
            )
            
            # compute reward
            reward = self._compute_reward(offspring, parent1, parent2, reward_type)
            batch_rewards.append(reward)
            total_reward += reward
            
            # track fitness
            offspring_fitness = self.fitness_function(offspring)
            fitness_values.append(offspring_fitness)
        
        # compute baseline
        if self.baseline_type == "batch_mean":
            baseline = np.mean(batch_rewards)
        elif self.baseline_type == "moving_average":
            baseline = self.baseline
        else:
            baseline = 0.0
        
        # compute policy gradient loss for the batch
        batch_loss = 0.0
        for i, (parent1, parent2) in enumerate(parent_pairs):
            # forward pass again for gradient computation
            offspring, log_probs, _ = self._forward_with_log_probs(
                parent1, parent2, temperature
            )
            
            # advantage = reward - baseline
            advantage = batch_rewards[i] - baseline
            
            # REINFORCE loss: -log_prob * advantage
            # sum over sequence length
            loss = -torch.sum(log_probs * advantage)
            batch_loss += loss
        
        # average loss over batch
        batch_loss = batch_loss / len(parent_pairs)
        
        # backpropagation
        self.optimizer.zero_grad()
        batch_loss.backward()
        
        # gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(
            list(self.operator.encoder.parameters()) + 
            list(self.operator.decoder.parameters()),
            max_norm=1.0
        )
        
        self.optimizer.step()
        
        # update baseline
        if self.baseline_type == "moving_average":
            self._update_baseline(np.mean(batch_rewards))
        
        self.operator.eval_mode()
        
        return {
            'loss': batch_loss.item(),
            'avg_reward': np.mean(batch_rewards),
            'avg_fitness': np.mean(fitness_values),
            'best_fitness': np.max(fitness_values),
            'baseline': baseline
        }
    
    def train(
        self,
        population_generator: Callable,
        num_episodes: int,
        batch_size: int = 32,
        temperature: float = 1.0,
        reward_type: str = "improvement",
        save_interval: int = 100,
        save_path: str = "models/dnc_checkpoint.pth",
        verbose: bool = True
    ):
        """
        Main training loop.
        
        Args:
            population_generator: Function that generates list of (parent1, parent2) tuples
            num_episodes: Number of training episodes
            batch_size: Number of parent pairs per episode
            temperature: Temperature for sampling
            reward_type: Type of reward computation
            save_interval: Save model every N episodes
            save_path: Path to save model checkpoints
            verbose: Whether to print training progress
        """
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        
        if verbose:
            pbar = tqdm(range(num_episodes), desc="Training DL Operator")
        else:
            pbar = range(num_episodes)
        
        for episode in pbar:
            # generate parent pairs
            parent_pairs = population_generator(batch_size)
            
            # train on batch
            stats = self.train_episode(
                parent_pairs,
                temperature=temperature,
                reward_type=reward_type
            )
            
            # update history
            self.history['losses'].append(stats['loss'])
            self.history['rewards'].append(stats['avg_reward'])
            self.history['avg_fitness'].append(stats['avg_fitness'])
            self.history['best_fitness'].append(stats['best_fitness'])
            
            # update progress bar
            if verbose:
                pbar.set_postfix({
                    'loss': f"{stats['loss']:.4f}",
                    'reward': f"{stats['avg_reward']:.4f}",
                    'fitness': f"{stats['avg_fitness']:.4f}",
                    'baseline': f"{stats['baseline']:.4f}"
                })
            
            # save checkpoint
            if (episode + 1) % save_interval == 0:
                checkpoint_path = save_path.replace('.pth', f'_ep{episode+1}.pth')
                self.save_checkpoint(checkpoint_path, episode)
                if verbose:
                    print(f"\nCheckpoint saved to {checkpoint_path}")
        
        # save final model
        self.save_checkpoint(save_path, num_episodes)
        if verbose:
            print(f"\nFinal model saved to {save_path}")
    
    def save_checkpoint(self, path: str, episode: int):
        """Saves training checkpoint"""
        torch.save({
            'episode': episode,
            'encoder': self.operator.encoder.state_dict(),
            'decoder': self.operator.decoder.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'history': self.history,
            'baseline': self.baseline,
            'gene_size': self.operator.gene_size,
            'hidden_size': self.operator.hidden_size
        }, path)
    
    def load_checkpoint(self, path: str):
        """Loads training checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.operator.encoder.load_state_dict(checkpoint['encoder'])
        self.operator.decoder.load_state_dict(checkpoint['decoder'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.history = checkpoint['history']
        self.baseline = checkpoint['baseline']
        return checkpoint['episode']
