from abc import ABC, abstractmethod


class BaseProblem(ABC):
    """Abstract base class for optimization problems"""
    
    @abstractmethod
    def fitness_function(self, individual: list) -> float:
        """
        Evaluates the fitness of an individual.
        
        Args:
            individual: Chromosome to evaluate
            
        Returns:
            Fitness score (higher is better)
        """
        pass
    
    @abstractmethod
    def generate_individual(self) -> list:
        """
        Generates a random individual for the population.
        
        Returns:
            Random chromosome
        """
        pass
    
    @abstractmethod
    def problem_name(self) -> str:
        """Returns the name of the problem"""
        pass
    
    def generate_population(self, size: int) -> list:
        """
        Generates initial population.
        
        Args:
            size: Population size
            
        Returns:
            List of individuals
        """
        return [self.generate_individual() for _ in range(size)]
    
    def __str__(self) -> str:
        return self.problem_name()
