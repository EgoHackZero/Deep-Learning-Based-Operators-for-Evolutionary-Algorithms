# from my old project - https://github.com/Mruzik1/Genetic-Algorithms/tree/main (modified for the assignment)
# Uses classes defined in other files for performing crossovers and mutations.
# The class itself contains methods for initializing the population, performing selection, and a fitness function.

import random
from typing import Callable, Optional

from genetic_algorithm.crossovers.base_crossover import BaseCrossover
from genetic_algorithm.mutations.base_mutation import BaseMutation


class GeneticAlgorithm:
    def __init__(
        self,
        crossover: BaseCrossover,
        mutation: BaseMutation,
        fitness_function: Callable,
        seed: Optional[int] = None
    ):
        if seed is not None:
            random.seed(seed)
        self.__crossover = crossover
        self.__mutation = mutation
        self.__fit_func = fitness_function
        self.__population = []
        self.__pop_history = {"mean": [], "max": []}
        self.__pop_size = 0

    # initializes the population
    def init_population(self, initial_pop: list):
        self.__population = initial_pop.copy()
        self.__pop_size = len(initial_pop)

    # population getter
    @property
    def population(self) -> list:
        return self.__population.copy()
    
    # population size getter
    @property
    def population_size(self) -> int:
        return self.__pop_size

    # "selector" param determines how many individuals with worst scores should be removed from the population
    def __replacement(self, selector: int):
        self.__population.sort(key=lambda x: self.__fit_func(x))
        self.__population = self.__population[selector:]

    # returns a list of every element"s fitness
    def __get_fitness_scores(self) -> list:
        return [self.__fit_func(i) for i in self.__population]

    # selects parents using tournament selection
    def __select_parents(self, tournament_size: int = 2) -> list:
        def tournament():
            participants = random.sample(self.__population, tournament_size)
            return max(participants, key=self.__fit_func)

        return [tournament(), tournament()]

    # performs crossover k-times, generates offsprings
    def __perform_crossover(self, k: int, **crossover_kwargs):
        for _ in range(k):
            parent1, parent2 = self.__select_parents()
            yield self.__crossover.perform(parent1, parent2, **crossover_kwargs)

    # mutates offsprings with a given probability
    def __perform_mutation(self, offsprings):
        for offspring in offsprings:
            yield self.__mutation.perform(offspring)

    # prints info about the current population
    def __print_info(self, population_num: int, k: int):
        curr_fitness = self.__get_fitness_scores()
        curr_mean = sum(curr_fitness) / self.__pop_size
        current_max = max(curr_fitness)

        print(f"Population: {population_num} / {k}\t \
                Current maximum: {current_max:.6f} \
                Current mean: {curr_mean:.6f}", end="\r")

    # population history getter
    # pop_history is a list of all mean population scores acquired during the algorithm"s run
    @property
    def history(self) -> list:
        return self.__pop_history

    # returns population's mean fitness score
    def __get_mean(self):
        return sum(self.__get_fitness_scores()) / self.__pop_size
    
    # save population
    def save_population(self, filename: str, serializer: Optional[Callable] = None):
        with open(filename, "w") as f:
            for individual in self.__population:
                if serializer:
                    f.write(f"{serializer(individual)}\n")
                else:
                    f.write(f"{' '.join(map(str, individual))}\n")

    # load population
    def load_population(self, filename: str, deserializer: Optional[Callable] = None):
        with open(filename, "r") as f:
            if deserializer:
                self.__population = [deserializer(line.strip()) for line in f.readlines()]
            else:
                self.__population = [line.strip().split() for line in f.readlines()]
        self.__pop_size = len(self.__population)

    # selector determines a number of individuals that will be replaced by offsprings
    # returns the best individual
    def start(
        self, 
        generations: int, 
        selector: int,
        save_interval: Optional[int] = None,
        save_filename: str = "population.txt",
        **crossover_kwargs
    ) -> list:
        if selector >= self.__pop_size:
            raise ValueError(f"Selector should be less than the population size: {selector} >= {self.__pop_size}")

        for i in range(generations):
            if generations >= 5:
                self.__print_info(i, generations)

            self.__replacement(selector)

            offsprings = self.__perform_crossover(self.__pop_size-len(self.__population), **crossover_kwargs)
            offsprings = self.__perform_mutation(offsprings)

            self.__population += list(offsprings)
            self.__pop_history["mean"].append(self.__get_mean())
            self.__pop_history["max"].append(max(self.__get_fitness_scores()))

            if save_interval and i % save_interval == 0:
                self.save_population(save_filename)
        
        return max(self.__population, key=lambda x: self.__fit_func(x))
    

if __name__ == "__main__":
    pass  # example usage
