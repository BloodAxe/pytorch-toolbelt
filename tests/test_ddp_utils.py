import numpy as np

from pytorch_toolbelt.utils.bucket_assignment import (
    naive_bucket_assignment,
    compute_bucket_imbalance_score,
    random_bucket_assignment,
    filler_bucket_assignment,
)


def test_approximate_bucket_assignment():
    cost = np.concatenate(
        [
            np.random.randint(0, 100, size=100),
            np.random.randint(100, 1000, size=50),
            np.random.randint(1000, 5000, size=5),
            np.random.randint(10000, 20000, size=1),
            np.random.randint(20000, 100000, size=3),
        ]
    )
    cost = np.random.permutation(cost)

    assignment = naive_bucket_assignment(cost, 4)

    print("naive_bucket_assignment  ", compute_bucket_imbalance_score(cost, assignment))
    print(np.bincount(assignment))

    assignment = random_bucket_assignment(cost, 4, 1000)

    print("random_bucket_assignment ", compute_bucket_imbalance_score(cost, assignment))
    print(np.bincount(assignment))

    assignment = filler_bucket_assignment(cost, 4)

    print("filler_bucket_assignment ", compute_bucket_imbalance_score(cost, assignment))
    print(np.bincount(assignment))
