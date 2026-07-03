import json
import requests
from pathlib import Path

# ============================================================
# Configuration
# ============================================================

BASE_URL = "https://gathering-hamper-reviving.ngrok-free.dev"

TOP_K = 10

TEST_FILE = Path("evaluation/test_queries.json")


# ============================================================
# Metrics
# ============================================================

def precision_at_k(predicted, expected, k):
    predicted = predicted[:k]

    hits = sum(
        item in expected
        for item in predicted
    )

    return hits / k


def recall_at_k(predicted, expected, k):
    predicted = predicted[:k]

    hits = sum(
        item in expected
        for item in predicted
    )

    if len(expected) == 0:
        return 0

    return hits / len(expected)


def hit_rate(predicted, expected):
    for item in predicted:
        if item in expected:
            return 1

    return 0


def reciprocal_rank(predicted, expected):
    for i, item in enumerate(predicted, start=1):
        if item in expected:
            return 1 / i

    return 0


def dcg(predicted, expected, k):
    score = 0

    for i, item in enumerate(predicted[:k], start=1):
        if item in expected:
            score += 1 / __import__("math").log2(i + 1)

    return score


def ndcg(predicted, expected, k):
    ideal = min(len(expected), k)

    if ideal == 0:
        return 0

    ideal_dcg = sum(
        1 / __import__("math").log2(i + 1)
        for i in range(1, ideal + 1)
    )

    return dcg(predicted, expected, k) / ideal_dcg


# ============================================================
# API Call
# ============================================================

def search(query):
    # -----------------------------
    # First turn
    # -----------------------------
    messages = [
        {
            "role": "user",
            "content": query,
        }
    ]

    response = requests.post(
        f"{BASE_URL}/chat",
        json={
            "messages": messages,
            "top_k": TOP_K,
        },
    )

    response.raise_for_status()

    result = response.json()

    # -----------------------------
    # Automatic clarification
    # -----------------------------
    if not result.get("end_of_conversation", True):

        print("\nAssistant:")
        print(result["reply"])

        messages.append(
            {
                "role": "assistant",
                "content": result["reply"],
            }
        )

        # automatic reply
        messages.append(
            {
                "role": "user",
                "content": "Mid",
            }
        )

        response = requests.post(
            f"{BASE_URL}/chat",
            json={
                "messages": messages,
                "top_k": TOP_K,
            },
        )

        response.raise_for_status()

        result = response.json()

    print("=" * 80)
    print("QUERY:", query)
    print(json.dumps(result, indent=2))

    return [
        item["name"]
        for item in result["recommendations"]
    ]


# ============================================================
# Evaluation
# ============================================================

def evaluate():
    with open(TEST_FILE) as f:
        tests = json.load(f)

    precision_scores = []
    recall_scores = []
    hit_scores = []
    mrr_scores = []
    ndcg_scores = []

    print("=" * 80)

    for sample in tests:
        query = sample["query"]
        expected = sample["expected"]

        predicted = search(query)

        precision = precision_at_k(
            predicted,
            expected,
            TOP_K,
        )

        recall = recall_at_k(
            predicted,
            expected,
            TOP_K,
        )

        hit = hit_rate(
            predicted,
            expected,
        )

        rr = reciprocal_rank(
            predicted,
            expected,
        )

        ndcg_score = ndcg(
            predicted,
            expected,
            TOP_K,
        )

        precision_scores.append(precision)
        recall_scores.append(recall)
        hit_scores.append(hit)
        mrr_scores.append(rr)
        ndcg_scores.append(ndcg_score)

        print(f"\nQuery: {query}")
        print("-" * 80)

        print("Expected:")
        print(expected)

        print()

        print("Predicted:")
        print(predicted)

        print()

        print(
            f"P@{TOP_K}: {precision:.3f} | "
            f"Recall: {recall:.3f} | "
            f"Hit: {hit} | "
            f"MRR: {rr:.3f} | "
            f"NDCG: {ndcg_score:.3f}"
        )

    print("\n" + "=" * 80)

    print("\nFINAL RESULTS\n")

    print(f"Precision@{TOP_K}: {sum(precision_scores)/len(precision_scores):.4f}")
    print(f"Recall@{TOP_K}:    {sum(recall_scores)/len(recall_scores):.4f}")
    print(f"Hit Rate:          {sum(hit_scores)/len(hit_scores):.4f}")
    print(f"MRR:               {sum(mrr_scores)/len(mrr_scores):.4f}")
    print(f"NDCG@{TOP_K}:      {sum(ndcg_scores)/len(ndcg_scores):.4f}")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    evaluate()
