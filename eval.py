"""
Evaluation framework using pre-computed proxy ground truth labels.
Computes NDCG@k, MAP, P@k for one or more ranking files.

Usage:
    python eval.py --rankings submission.csv
    python eval.py --live  # run rank.py then evaluate
"""
import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path

BASE = Path(__file__).parent
GT_PATH = BASE / "gt_labels.json"


def load_gt():
    with open(GT_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    relevance = {k: int(v) for k, v in data['relevance'].items()}
    return relevance


def dcg(rel_scores):
    return sum(rel / math.log2(i + 2) for i, rel in enumerate(rel_scores))


def idcg(rel_scores):
    return dcg(sorted(rel_scores, reverse=True))


def compute_ndcg(ranked_ids, relevance_map, k):
    rel = [relevance_map.get(cid, 0) for cid in ranked_ids[:k]]
    actual = dcg(rel)
    ideal_vals = sorted(relevance_map.values(), reverse=True)[:k]
    ideal = dcg(ideal_vals)
    return actual / ideal if ideal > 0 else 0.0


def compute_map(ranked_ids, relevance_map):
    relevant = [cid for cid, t in relevance_map.items() if t >= 3]
    if not relevant:
        return 0.0
    precisions = []
    correct = 0
    for i, cid in enumerate(ranked_ids):
        if cid in relevant:
            correct += 1
            precisions.append(correct / (i + 1))
    return sum(precisions) / len(relevant) if precisions else 0.0


def compute_p_at_k(ranked_ids, relevance_map, k):
    relevant = {cid for cid, t in relevance_map.items() if t >= 3}
    count = sum(1 for cid in ranked_ids[:k] if cid in relevant)
    return count / k


def evaluate_ranking(ranked_ids, relevance_map):
    return {
        'NDCG@10': compute_ndcg(ranked_ids, relevance_map, 10),
        'NDCG@50': compute_ndcg(ranked_ids, relevance_map, 50),
        'NDCG@100': compute_ndcg(ranked_ids, relevance_map, 100),
        'MAP': compute_map(ranked_ids, relevance_map),
        'P@5': compute_p_at_k(ranked_ids, relevance_map, 5),
        'P@10': compute_p_at_k(ranked_ids, relevance_map, 10),
        'composite': (
            0.50 * compute_ndcg(ranked_ids, relevance_map, 10) +
            0.30 * compute_ndcg(ranked_ids, relevance_map, 50) +
            0.15 * compute_map(ranked_ids, relevance_map) +
            0.05 * compute_p_at_k(ranked_ids, relevance_map, 10)
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rankings', nargs='*', help='CSV ranking files to evaluate')
    parser.add_argument('--live', action='store_true', help='Run rank.py then evaluate')
    args = parser.parse_args()

    if not GT_PATH.exists():
        print(f"ERROR: GT labels not found at {GT_PATH}. Run build_gt.py first.")
        sys.exit(1)

    relevance = load_gt()
    dist = {}
    for t in relevance.values():
        dist[t] = dist.get(t, 0) + 1
    print(f"Proxy GT ({len(relevance)} labels): {dict(sorted(dist.items()))}")
    print(f"  Tier 3+ count: {sum(1 for t in relevance.values() if t >= 3)}")
    print()

    rankings_to_eval = list(args.rankings or [])

    if args.live:
        print("Running rank.py...")
        subprocess.run([sys.executable, str(BASE / "rank.py"),
                       "--candidates", str(BASE / "candidates.jsonl"),
                       "--out", str(BASE / "_eval_submission.csv")], check=True)
        rankings_to_eval.append(str(BASE / "_eval_submission.csv"))

    if not rankings_to_eval:
        parser.print_help()
        return

    for rpath in rankings_to_eval:
        print(f"{'='*60}")
        print(f"Evaluating: {rpath}")
        print('='*60)

        with open(rpath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            ranked_ids = [row['candidate_id'] for row in reader]

        overlap = sum(1 for cid in ranked_ids[:100] if cid in relevance)
        print(f"Overlap with GT in top 100: {overlap}/{len(relevance)}")

        honeypots = sum(1 for cid in ranked_ids[:100] if relevance.get(cid) == 0)
        print(f"Honeypots in top 100: {honeypots}")

        tier_counts = {}
        for cid in ranked_ids[:100]:
            t = relevance.get(cid, -1)
            tier_counts[t] = tier_counts.get(t, 0) + 1
        unlabeled = tier_counts.pop(-1, 0)
        print(f"Tier distribution in top 100: {dict(sorted(tier_counts.items()))}", end="")
        if unlabeled:
            print(f" (+{unlabeled} unlabeled)")
        else:
            print()

        metrics = evaluate_ranking(ranked_ids, relevance)
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")
        print()


if __name__ == '__main__':
    main()
