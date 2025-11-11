import random
import math
from problems.base_problem import BaseProblem


class TSPProblem(BaseProblem):
    def __init__(
        self, 
        num_cities: int = 10,
        seed: int = None
    ):
        """
        TSP problem initialization.
        
        Args:
            num_cities: Number of cities
            seed: Random seed for city generation
        """
        self.num_cities = num_cities
        
        # generate random city coordinates
        if seed is not None:
            random.seed(seed)
        
        self.cities = []
        for _ in range(num_cities):
            x = random.uniform(0, 100)
            y = random.uniform(0, 100)
            self.cities.append((x, y))
    
    def _distance(self, city1_idx: int, city2_idx: int) -> float:
        """Calculate Euclidean distance between two cities"""
        x1, y1 = self.cities[city1_idx]
        x2, y2 = self.cities[city2_idx]
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    
    def fitness_function(self, individual: list) -> float:
        """
        Calculate total tour distance (minimize distance = maximize negative distance).
        Individual is a permutation of city indices.
        """
        total_distance = 0
        
        # calculate distance for each consecutive pair
        for i in range(len(individual) - 1):
            total_distance += self._distance(individual[i], individual[i + 1])
        
        # add distance from last city back to first
        total_distance += self._distance(individual[-1], individual[0])
        
        # return negative distance (to maximize)
        return -total_distance
    
    def generate_individual(self) -> list:
        """Generate random permutation of cities"""
        individual = list(range(self.num_cities))
        random.shuffle(individual)
        return individual
    
    def problem_name(self) -> str:
        return f"TSP (cities={self.num_cities})"
