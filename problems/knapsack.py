import random
from problems.base_problem import BaseProblem


class KnapsackProblem(BaseProblem):
    def __init__(
        self, 
        num_items: int = 20,
        capacity: int = 100,
        seed: int = None
    ):
        """
        Knapsack problem initialization.
        
        Args:
            num_items: Number of items available
            capacity: Weight capacity of knapsack
            seed: Random seed for item generation
        """
        self.num_items = num_items
        self.capacity = capacity
        
        # generate random items (weight, value)
        if seed is not None:
            random.seed(seed)
        
        self.items = []
        for _ in range(num_items):
            weight = random.randint(1, 30)
            value = random.randint(1, 100)
            self.items.append((weight, value))
    
    def fitness_function(self, individual: list) -> float:
        """
        Calculate total value if within capacity, penalize if over.
        Individual is binary: 1 = take item, 0 = leave item
        """
        total_weight = 0
        total_value = 0
        
        for i, selected in enumerate(individual):
            if selected:
                total_weight += self.items[i][0]
                total_value += self.items[i][1]
        
        # penalty for exceeding capacity
        if total_weight > self.capacity:
            return max(0, total_value - (total_weight - self.capacity) * 10)
        
        return total_value
    
    def generate_individual(self) -> list:
        """Generate random binary selection"""
        return [random.randint(0, 1) for _ in range(self.num_items)]
    
    def problem_name(self) -> str:
        return f"Knapsack (items={self.num_items}, capacity={self.capacity})"
