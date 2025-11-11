from random import randint
from genetic_algorithm.crossovers.base_crossover import BaseCrossover


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
