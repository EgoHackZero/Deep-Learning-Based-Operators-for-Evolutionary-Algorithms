from abc import ABC, abstractmethod
from random import random
from typing import Optional


# abstract base class for all mutation operators
class BaseMutation(ABC):
    def __init__(self, chance: float = 0.1, seed: Optional[int] = None):
        if not 0 <= chance <= 1:
            raise ValueError(f"Mutation chance must be between 0 and 1, got {chance}")
        self._chance = chance
        if seed is not None:
            from random import seed as set_seed
            set_seed(seed)
    
    @abstractmethod
    def _mutate(self, offspring: list, **kwargs) -> list:
        """
        Performs the actual mutation operation.
        
        Args:
            offspring: Individual to mutate
            **kwargs: Additional parameters specific to the mutation type
            
        Returns:
            Mutated offspring as a list
        """
        pass
    
    def perform(self, offspring: list, **kwargs) -> list:
        """
        Performs mutation with probability check.
        
        Args:
            offspring: Individual to mutate
            **kwargs: Additional parameters specific to the mutation type
            
        Returns:
            Mutated or original offspring as a list
        """
        if random() > self._chance:
            return offspring
        return self._mutate(offspring, **kwargs)
    
    def _validate_offspring(self, offspring: list, min_length: int = 2):
        """Validates that offspring has minimum required length"""
        if len(offspring) < min_length:
            raise ValueError(f"Offspring length must be at least {min_length}, got {len(offspring)}")
