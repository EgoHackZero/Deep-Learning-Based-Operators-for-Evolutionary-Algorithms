from random import randint
from typing import Optional
from genetic_algorithm.crossovers.base_crossover import BaseCrossover


# single-point crossover
class SinglePointCrossover(BaseCrossover):
    def perform(self, parent1: list, parent2: list, **kwargs) -> list:
        self._validate_parents(parent1, parent2)
        point = randint(1, len(parent1) - 1)
        return parent1[:point] + parent2[point:]


# multi-point crossover
class MultiPointCrossover(BaseCrossover):
    def __init__(self, points: int = 2, seed: Optional[int] = None):
        super().__init__(seed)
        if points <= 0:
            raise ValueError("Number of points must be greater than 0")
        self.__points = points
    
    def perform(self, parent1: list, parent2: list, **kwargs) -> list:
        self._validate_parents(parent1, parent2)
        points = kwargs.get('points', self.__points)
        
        prev_p = 0
        offspring = []

        for i in range(points):
            p = randint(prev_p+1, len(parent1)-points+i)
            offspring += parent2[prev_p:p] if i%2 else parent1[prev_p:p]
            prev_p = p

        offspring += parent2[prev_p:] if points%2 else parent1[prev_p:]
        return offspring


# uniform crossover
class UniformCrossover(BaseCrossover):
    def perform(self, parent1: list, parent2: list, **kwargs) -> list:
        self._validate_parents(parent1, parent2)
        mask = [randint(0, 1) for _ in range(len(parent1))]
        return [parent2[i] if mask[i] else parent1[i] for i in range(len(parent1))]
