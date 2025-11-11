from abc import ABC, abstractmethod
from typing import Optional


# abstract base class for all crossover operators
class BaseCrossover(ABC):
    def __init__(self, seed: Optional[int] = None):
        if seed is not None:
            from random import seed as set_seed
            set_seed(seed)
    
    @abstractmethod
    def perform(self, parent1: list, parent2: list, **kwargs) -> list:
        """
        Performs crossover on two parents and returns offspring.
        
        Args:
            parent1: First parent
            parent2: Second parent
            **kwargs: Additional parameters specific to the crossover type
            
        Returns:
            Offspring as a list
        """
        pass
    
    def _validate_parents(self, parent1: list, parent2: list):
        """Validates that parents have the same length"""
        if len(parent1) != len(parent2):
            raise ValueError(f"Parents' length should be the same! You have: {len(parent1)} and {len(parent2)}")