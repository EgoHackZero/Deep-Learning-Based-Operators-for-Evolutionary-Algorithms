# from my old project - https://github.com/Mruzik1/Genetic-Algorithms/tree/main

from abc import ABC, abstractmethod
from random import randint
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


# cycle crossover for permutation-based problems
class CycleCrossover(BaseCrossover):
    def perform(self, parent1: list, parent2: list, **kwargs) -> list:
        self._validate_parents(parent1, parent2)
        used_genes = []
        cycles = []

        for idx, gene in enumerate(parent1):
            if gene not in used_genes:
                cycle = self.__generate_cycle(parent1, parent2, idx)
                cycles.append(cycle)
                used_genes += cycle
        
        return [g2 if self.__is_odd_cycle(g1, cycles) else g1 for g1, g2 in zip(parent1, parent2)]
    
    def __generate_cycle(self, p1: list, p2: list, idx: int) -> list:
        result = [p1[idx], p2[idx]]

        while result[-1] != result[0]:
            result.append(p2[p1.index(result[-1])])

        return result[:-1]

    def __is_odd_cycle(self, gene, cycles: list) -> bool:
        for i, cycle in enumerate(cycles):
            if gene in cycle:
                return bool(i%2)


# partially mapped crossover for permutation-based problems
class PMXCrossover(BaseCrossover):
    def perform(self, parent1: list, parent2: list, **kwargs) -> list:
        self._validate_parents(parent1, parent2)
        start, end = self.__find_subset(len(parent1))
        offspring = [0]*start + parent1[start:end] + [0]*len(parent1[end:])

        for i, gene in enumerate(parent2):
            new_idx = i

            if gene in offspring:
                continue
            elif i in range(start, end):
                new_idx = self.__place_from_subset(parent1, parent2, i, (start, end))

            offspring[new_idx] = gene
        
        return offspring
    
    def __find_subset(self, length: int) -> tuple:
        start = randint(0, length-2)
        end = randint(start+2, length)
        return start, end

    def __place_from_subset(self, p1: list, p2: list, idx: int, subset: tuple) -> int:
        new_idx = p2.index(p1[idx])

        if new_idx in range(*subset):
            return self.__place_from_subset(p1, p2, new_idx, subset)
        return new_idx
