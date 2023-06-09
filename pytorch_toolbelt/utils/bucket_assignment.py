import numpy as np


def naive_bucket_assignment(costs: np.ndarray, num_buckets: int) -> np.ndarray:
    ordered_indexes = np.argsort(costs)
    assignment = ordered_indexes % num_buckets
    return assignment


def compute_bucket_imbalance_score(costs, assignment):
    buckets = np.unique(assignment)
    costs_per_bucket = []
    for bucket in buckets:
        costs_per_bucket.append(np.sum(costs[assignment == bucket]))
    return np.std(costs_per_bucket)


def random_bucket_assignment(costs: np.ndarray, num_buckets: int, max_iterations: int) -> np.ndarray:
    best_indexes = naive_bucket_assignment(costs, num_buckets)
    best_cost = compute_bucket_imbalance_score(costs, best_indexes)

    for _ in range(max_iterations):
        new_indexes = np.random.permutation(best_indexes)
        new_cost = compute_bucket_imbalance_score(costs, new_indexes)
        if new_cost < best_cost:
            best_indexes = new_indexes
            best_cost = new_cost

    return best_indexes


def filler_bucket_assignment(costs: np.ndarray, num_buckets: int) -> np.ndarray:
    order = np.argsort(-costs)
    current_buckets_cost = np.zeros(num_buckets)
    assignment = np.zeros_like(costs, dtype=int)
    for element_index in order:
        bucket_with_min_cost = np.argmin(current_buckets_cost)
        assignment[element_index] = bucket_with_min_cost
        current_buckets_cost[bucket_with_min_cost] += costs[element_index]

    return assignment
