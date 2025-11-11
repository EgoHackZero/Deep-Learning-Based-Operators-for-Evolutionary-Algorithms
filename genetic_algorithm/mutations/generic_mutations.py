from random import randint
from genetic_algorithm.mutations.base_mutation import BaseMutation


# inversion mutation - reverses a random subset
class InversionMutation(BaseMutation):
    def _mutate(self, offspring: list, **kwargs) -> list:
        self._validate_offspring(offspring)
        tmp_offspring = offspring.copy()
        start, end = self.__find_subset(len(tmp_offspring))
        tmp_offspring = tmp_offspring[:start] + list(reversed(tmp_offspring[start:end])) + tmp_offspring[end:]
        return tmp_offspring
    
    def __find_subset(self, length: int) -> tuple:
        start = randint(0, length-2)
        end = randint(start+2, length)
        return start, end


# swap mutation - switches 2 random elements
class SwapMutation(BaseMutation):
    def _mutate(self, offspring: list, **kwargs) -> list:
        self._validate_offspring(offspring)
        tmp_offspring = offspring.copy()
        first = randint(0, len(tmp_offspring)-1)
        second = first
        while second == first:
            second = randint(0, len(tmp_offspring)-1)
        tmp_offspring[first], tmp_offspring[second] = tmp_offspring[second], tmp_offspring[first]
        return tmp_offspring