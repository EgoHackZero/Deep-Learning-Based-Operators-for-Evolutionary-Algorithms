import random
from problems.base_problem import BaseProblem


class OneMaxProblem(BaseProblem):
    def __init__(self, chromosome_length: int = 50):
        """
        OneMax problem initialization.
        
        Args:
            chromosome_length: Length of binary string
        """
        self.chromosome_length = chromosome_length
    
    def fitness_function(self, individual: list) -> float:
        """Count the number of 1s"""
        return sum(individual)
    
    def generate_individual(self) -> list:
        """Generate random binary string"""
        return [random.randint(0, 1) for _ in range(self.chromosome_length)]
    
    def problem_name(self) -> str:
        return f"OneMax (length={self.chromosome_length})"
