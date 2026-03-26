"""
Step 13 — Evaluation
====================
Measures:
  1. Domain Classification Accuracy
  2. Confidence Score Distribution
  3. Hallucination Rate (keyword check)
  4. Response Time
  5. Low Confidence Rate per domain

Run:
  python evaluation.py
"""

import json
import time
import os
from langgraph_orchestration import run_query


# ============================================================
# Test set — ground truth domain labels
# ============================================================
TEST_QUERIES = [
    # (query, expected_domain)
    ("What is polymorphism in Python?",                  "coding"),
    ("How do I implement a binary search tree?",         "coding"),
    ("What is a REST API?",                              "coding"),
    ("What is Article 21 of the Indian Constitution?",   "legal"),
    ("Explain IPC section 302",                          "legal"),
    ("What are fundamental rights in India?",            "legal"),
    ("What are the symptoms of Type 2 Diabetes?",        "medical"),
    ("How does chemotherapy work?",                      "medical"),
    ("What is hypertension?",                            "medical"),
    ("What is insider trading?",                         "finance"),
    ("Explain compound interest",                        "finance"),
    ("What is a mutual fund?",                           "finance"),
]

# Keywords that suggest hallucination / garbage output
HALLUCINATION_SIGNALS = [
    "weil weil", "Gö Gö", "Liege Liege",
    "universitaire", "I don't know", "I cannot",
    "as an AI", "I'm not able"
]


# ============================================================
# Run evaluation
# ============================================================
def evaluate():

    print("=" * 60)
    print("EVALUATION — AI Knowledge Fabric")
    print("=" * 60)

    results = []

    for query, expected_domain in TEST_QUERIES:

        print(f"\nQuery: {query[:50]}...")

        start_time = time.time()
        result     = run_query(query)
        elapsed    = round(time.time() - start_time, 2)

        domain_correct = result["domain"] == expected_domain

        # Check for hallucination signals in answer
        answer_lower   = result["answer"].lower()
        hallucinated   = any(sig.lower() in answer_lower for sig in HALLUCINATION_SIGNALS)

        entry = {
            "query":            query,
            "expected_domain":  expected_domain,
            "predicted_domain": result["domain"],
            "domain_correct":   domain_correct,
            "confidence":       result["agent_confidence"],
            "low_confidence":   result["low_confidence"],
            "hallucinated":     hallucinated,
            "response_time_s":  elapsed,
        }

        results.append(entry)

        status = "✓" if domain_correct else "✗"
        print(f"  {status} Domain: {result['domain']} (expected: {expected_domain})")
        print(f"  Confidence : {result['agent_confidence']}")
        print(f"  Time       : {elapsed}s")
        print(f"  Hallucinated: {hallucinated}")


    # ============================================================
    # Compute metrics
    # ============================================================
    total = len(results)

    domain_accuracy   = sum(1 for r in results if r["domain_correct"])    / total
    hallucination_rate= sum(1 for r in results if r["hallucinated"])       / total
    low_conf_rate     = sum(1 for r in results if r["low_confidence"])     / total
    avg_confidence    = sum(r["confidence"] for r in results)              / total
    avg_response_time = sum(r["response_time_s"] for r in results)         / total

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Total queries            : {total}")
    print(f"Domain Classification    : {round(domain_accuracy * 100, 1)}%")
    print(f"Avg Confidence           : {round(avg_confidence * 100, 1)}%")
    print(f"Low Confidence Rate      : {round(low_conf_rate * 100, 1)}%")
    print(f"Hallucination Rate       : {round(hallucination_rate * 100, 1)}%")
    print(f"Avg Response Time        : {round(avg_response_time, 2)}s")

    # Per-domain breakdown
    print("\nPer-domain accuracy:")
    for domain in ["coding", "legal", "medical", "finance"]:
        domain_results = [r for r in results if r["expected_domain"] == domain]
        correct        = sum(1 for r in domain_results if r["domain_correct"])
        print(f"  {domain:10s}: {correct}/{len(domain_results)}")

    # Save results
    with open("evaluation_results.json", "w") as f:
        json.dump({
            "metrics": {
                "domain_accuracy":   domain_accuracy,
                "avg_confidence":    avg_confidence,
                "low_conf_rate":     low_conf_rate,
                "hallucination_rate":hallucination_rate,
                "avg_response_time": avg_response_time,
            },
            "per_query": results
        }, f, indent=2)

    print("\nSaved to evaluation_results.json")


if __name__ == "__main__":
    evaluate()