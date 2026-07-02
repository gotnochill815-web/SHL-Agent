import numpy as np


def precision_at_k(predicted, expected, k=5):
    predicted = predicted[:k]

    hits = len(set(predicted) & set(expected))

    return hits / k


def recall_at_k(predicted, expected, k=5):
    predicted = predicted[:k]

    hits = len(set(predicted) & set(expected))

    return hits / len(expected)


def hit_rate(predicted, expected):
    return int(
        len(set(predicted) & set(expected)) > 0
    )


def reciprocal_rank(predicted, expected):

    for i, p in enumerate(predicted):

        if p in expected:
            return 1 / (i + 1)

    return 0


def ndcg(predicted, expected, k=5):

    predicted = predicted[:k]

    dcg = 0

    for i, p in enumerate(predicted):

        if p in expected:

            dcg += 1 / np.log2(i + 2)

    ideal = min(len(expected), k)

    idcg = sum(
        1 / np.log2(i + 2)
        for i in range(ideal)
    )

    return dcg / idcg if idcg > 0 else 0