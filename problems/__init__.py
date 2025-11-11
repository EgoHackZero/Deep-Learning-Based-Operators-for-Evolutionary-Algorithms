from problems.onemax import OneMaxProblem
from problems.knapsack import KnapsackProblem
from problems.tsp import TSPProblem


PROBLEMS = {
    'onemax': OneMaxProblem,
    'knapsack': KnapsackProblem,
    'tsp': TSPProblem
}


def get_problem(name: str, **kwargs):
    """
    Get problem instance by name.
    
    Args:
        name: Problem name ('onemax', 'knapsack', 'tsp')
        **kwargs: Problem-specific parameters
        
    Returns:
        Problem instance
    """
    if name.lower() not in PROBLEMS:
        raise ValueError(f"Unknown problem: {name}. Available: {list(PROBLEMS.keys())}")
    
    return PROBLEMS[name.lower()](**kwargs)
