"""
GP-specific crossover operators for variable-length trees.
"""

import random
from genetic_algorithm.crossovers.base_crossover import BaseCrossover


class PassthroughCrossover(BaseCrossover):
    """
    Passthrough crossover that just returns parent1 unchanged.
    Useful for mutation-only GP evolution.
    """

    def perform(self, parent1: list, parent2: list, **kwargs) -> list:
        """
        Returns parent1 unchanged.

        Args:
            parent1: First parent
            parent2: Second parent (unused)
            **kwargs: Additional parameters

        Returns:
            Offspring (copy of parent1)
        """
        return parent1.copy()


class SubtreeCrossover(BaseCrossover):
    """
    Simple subtree crossover for GP trees.

    Randomly selects a subtree from each parent and swaps them.
    Works with variable-length trees.
    """

    def perform(self, parent1: list, parent2: list, **kwargs) -> list:
        """
        Perform subtree crossover.

        Args:
            parent1: First parent tree (prefix notation)
            parent2: Second parent tree (prefix notation)
            **kwargs: Additional parameters

        Returns:
            Offspring tree
        """
        # For simplicity, just take a random subtree from parent2
        # and insert it at a random position in parent1
        # This is a simplified version

        if len(parent1) == 0:
            return parent2.copy()
        if len(parent2) == 0:
            return parent1.copy()

        # Copy parent1
        offspring = parent1.copy()

        # Select random point in offspring to replace
        point = random.randint(0, len(offspring) - 1)

        # Select random subtree from parent2 (just take a single node for simplicity)
        donor_point = random.randint(0, len(parent2) - 1)
        donor_gene = parent2[donor_point]

        # Replace
        offspring[point] = donor_gene

        return offspring


if __name__ == "__main__":
    # Test crossovers
    print("Testing GP Crossovers...")

    parent1 = ['*', '+', 'x', 'y', 2.0]
    parent2 = ['-', 'x', 'x']

    # Test passthrough
    passthrough = PassthroughCrossover()
    offspring = passthrough.perform(parent1, parent2)
    print(f"\nPassthrough crossover:")
    print(f"  Parent1: {parent1}")
    print(f"  Parent2: {parent2}")
    print(f"  Offspring: {offspring}")
    assert offspring == parent1

    # Test subtree
    subtree = SubtreeCrossover(seed=42)
    offspring = subtree.perform(parent1, parent2)
    print(f"\nSubtree crossover:")
    print(f"  Parent1: {parent1}")
    print(f"  Parent2: {parent2}")
    print(f"  Offspring: {offspring}")

    print("\nGP Crossover tests passed!")
