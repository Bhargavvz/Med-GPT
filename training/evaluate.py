#!/usr/bin/env python3
"""
MedGPT Evaluation Script
Computes accuracy, BLEU-1, ROUGE-L, and Token F1 on test data.
"""

import argparse
import json
import os
import re
import string
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import List, Tuple

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# =========================================================================
# Answer Normalization
# =========================================================================

# Synonym mappings
SYNONYM_MAP = {
    "y": "yes", "true": "yes", "positive": "yes", "correct": "yes",
    "n": "no", "false": "no", "negative": "no", "incorrect": "no",
    "0": "no", "1": "yes",
    "ct scan": "ct", "computed tomography": "ct",
    "mri scan": "mri", "magnetic resonance imaging": "mri",
    "x-ray": "xray", "x ray": "xray",
    "ultrasound": "us", "ultrasonography": "us",
}

ARTICLES = {"a", "an", "the"}
COMMON_PREFIXES = {"it is", "this is", "the answer is", "answer:", "answer is"}


def normalize_answer(answer: str) -> str:
    """
    Normalize an answer for evaluation.
    - Lowercase
    - Strip punctuation
    - Remove articles (a/an/the)
    - Map synonyms
    - Strip whitespace
    """
    if not answer:
        return ""

    # Lowercase
    ans = answer.lower().strip()

    # Remove common answer prefixes
    for prefix in COMMON_PREFIXES:
        if ans.startswith(prefix):
            ans = ans[len(prefix):].strip()

    # Remove punctuation
    ans = ans.translate(str.maketrans("", "", string.punctuation))

    # Split into words
    words = ans.split()

    # Remove articles
    words = [w for w in words if w not in ARTICLES]

    # Rejoin
    ans = " ".join(words).strip()

    # Map synonyms
    if ans in SYNONYM_MAP:
        ans = SYNONYM_MAP[ans]

    return ans


# =========================================================================
# Metrics
# =========================================================================

def compute_accuracy(preds: List[str], refs: List[str]) -> float:
    """Exact match accuracy after normalization."""
    if not preds:
        return 0.0
    correct = sum(
        1 for p, r in zip(preds, refs)
        if normalize_answer(p) == normalize_answer(r)
    )
    return correct / len(preds)


def compute_bleu1(preds: List[str], refs: List[str]) -> float:
    """BLEU-1 (unigram precision) averaged over all samples."""
    scores = []
    for pred, ref in zip(preds, refs):
        pred_tokens = normalize_answer(pred).split()
        ref_tokens = normalize_answer(ref).split()

        if not pred_tokens or not ref_tokens:
            scores.append(0.0)
            continue

        # Count unigram matches
        ref_counts = Counter(ref_tokens)
        pred_counts = Counter(pred_tokens)

        matches = 0
        for token, count in pred_counts.items():
            matches += min(count, ref_counts.get(token, 0))

        precision = matches / len(pred_tokens) if pred_tokens else 0
        scores.append(precision)

    return np.mean(scores) if scores else 0.0


def compute_rouge_l(preds: List[str], refs: List[str]) -> float:
    """ROUGE-L (longest common subsequence) averaged over all samples."""

    def lcs_length(x: List[str], y: List[str]) -> int:
        """Compute LCS length."""
        m, n = len(x), len(y)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if x[i - 1] == y[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        return dp[m][n]

    scores = []
    for pred, ref in zip(preds, refs):
        pred_tokens = normalize_answer(pred).split()
        ref_tokens = normalize_answer(ref).split()

        if not pred_tokens or not ref_tokens:
            scores.append(0.0)
            continue

        lcs = lcs_length(pred_tokens, ref_tokens)
        precision = lcs / len(pred_tokens) if pred_tokens else 0
        recall = lcs / len(ref_tokens) if ref_tokens else 0

        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0

        scores.append(f1)

    return np.mean(scores) if scores else 0.0


def compute_token_f1(preds: List[str], refs: List[str]) -> float:
    """Token-level F1 score averaged over all samples."""
    scores = []
    for pred, ref in zip(preds, refs):
        pred_tokens = normalize_answer(pred).split()
        ref_tokens = normalize_answer(ref).split()

        if not pred_tokens and not ref_tokens:
            scores.append(1.0)
            continue
        if not pred_tokens or not ref_tokens:
            scores.append(0.0)
            continue

        common = Counter(pred_tokens) & Counter(ref_tokens)
        n_common = sum(common.values())

        if n_common == 0:
            scores.append(0.0)
            continue

        precision = n_common / len(pred_tokens)
        recall = n_common / len(ref_tokens)
        f1 = 2 * precision * recall / (precision + recall)
        scores.append(f1)

    return np.mean(scores) if scores else 0.0


def compute_metrics_from_predictions(
    predictions: List[str],
    references: List[str],
    question_types: List[str] = None,
) -> dict:
    """
    Compute all metrics from lists of predictions and references.

    Returns:
        Dictionary with all metrics, including per-type breakdowns
    """
    metrics = {}

    # Overall metrics
    metrics["accuracy"] = compute_accuracy(predictions, references)
    metrics["bleu1"] = compute_bleu1(predictions, references)
    metrics["rouge_l"] = compute_rouge_l(predictions, references)
    metrics["token_f1"] = compute_token_f1(predictions, references)
    metrics["n_samples"] = len(predictions)

    # Per-type metrics if question types are provided
    if question_types:
        type_groups = defaultdict(lambda: ([], []))
        for pred, ref, qt in zip(predictions, references, question_types):
            type_groups[qt][0].append(pred)
            type_groups[qt][1].append(ref)

        for qt, (type_preds, type_refs) in type_groups.items():
            metrics[f"accuracy_{qt}"] = compute_accuracy(type_preds, type_refs)
            metrics[f"n_samples_{qt}"] = len(type_preds)

    return metrics


# =========================================================================
# Full Evaluation Pipeline
# =========================================================================

def run_full_evaluation(
    adapter_path: str,
    test_file: str,
    config_path: str = "configs/config.yaml",
    max_samples: int = None,
    output_file: str = None,
    verbose: bool = True,
):
    """
    Run full evaluation: load model, generate predictions, compute metrics.
    """
    from models.medgpt import MedGPT, load_config

    config = load_config(config_path)

    # Load model
    print(f"Loading model from: {adapter_path}")
    medgpt = MedGPT.from_adapter(adapter_path, config=config)

    # Load test data
    with open(test_file) as f:
        test_data = json.load(f)

    if max_samples:
        test_data = test_data[:max_samples]

    print(f"Evaluating on {len(test_data)} samples...")

    # Generate predictions
    predictions = []
    references = []
    question_types = []
    datasets = []
    results = []

    from tqdm import tqdm

    for i, sample in enumerate(tqdm(test_data, desc="Evaluating", ncols=80)):
        if not os.path.exists(sample["image_path"]):
            continue

        try:
            pred = medgpt.generate(
                image_path=sample["image_path"],
                question=sample["question"],
                knowledge=sample.get("knowledge", ""),
                max_new_tokens=64,
                do_sample=False,
            )
        except Exception as e:
            pred = ""

        predictions.append(pred)
        references.append(sample["answer"])
        question_types.append(sample.get("question_type", "open"))
        datasets.append(sample.get("dataset", "unknown"))

        results.append({
            "question": sample["question"],
            "prediction": pred,
            "prediction_normalized": normalize_answer(pred),
            "reference": sample["answer"],
            "reference_normalized": normalize_answer(sample["answer"]),
            "correct": normalize_answer(pred) == normalize_answer(sample["answer"]),
            "question_type": sample.get("question_type", "open"),
            "dataset": sample.get("dataset", "unknown"),
        })

        if verbose and i < 10:
            status = "✓" if results[-1]["correct"] else "✗"
            tqdm.write(f"  [{i}] {status} Q: {sample['question'][:50]}")
            tqdm.write(f"         Pred: {pred[:50]} | Ref: {sample['answer'][:50]}")

    # Compute metrics
    metrics = compute_metrics_from_predictions(predictions, references, question_types)

    # Per-dataset metrics
    dataset_groups = defaultdict(lambda: ([], [], []))
    for pred, ref, qt, ds in zip(predictions, references, question_types, datasets):
        dataset_groups[ds][0].append(pred)
        dataset_groups[ds][1].append(ref)
        dataset_groups[ds][2].append(qt)

    for ds_name, (ds_preds, ds_refs, ds_types) in dataset_groups.items():
        ds_metrics = compute_metrics_from_predictions(ds_preds, ds_refs, ds_types)
        for k, v in ds_metrics.items():
            metrics[f"{ds_name}/{k}"] = v

    # Print results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"\nOverall ({metrics['n_samples']} samples):")
    print(f"  Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.1f}%)")
    print(f"  BLEU-1:    {metrics['bleu1']:.4f}")
    print(f"  ROUGE-L:   {metrics['rouge_l']:.4f}")
    print(f"  Token F1:  {metrics['token_f1']:.4f}")

    if "accuracy_closed" in metrics:
        print(f"\nClosed (Yes/No) ({metrics.get('n_samples_closed', 0)} samples):")
        print(f"  Accuracy:  {metrics['accuracy_closed']:.4f} ({metrics['accuracy_closed']*100:.1f}%)")

    if "accuracy_open" in metrics:
        print(f"\nOpen-ended ({metrics.get('n_samples_open', 0)} samples):")
        print(f"  Accuracy:  {metrics['accuracy_open']:.4f} ({metrics['accuracy_open']*100:.1f}%)")

    # Per-dataset results
    for ds_name in sorted(dataset_groups.keys()):
        ds_acc = metrics.get(f"{ds_name}/accuracy", 0)
        ds_n = metrics.get(f"{ds_name}/n_samples", 0)
        print(f"\n{ds_name} ({ds_n} samples):")
        print(f"  Accuracy: {ds_acc:.4f} ({ds_acc*100:.1f}%)")

    # Save results
    if output_file:
        os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
        output = {
            "metrics": metrics,
            "predictions": results,
        }
        with open(output_file, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nDetailed results saved to: {output_file}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate MedGPT")
    parser.add_argument("--adapter_path", required=True, help="Path to LoRA adapter")
    parser.add_argument("--test_file", required=True, help="Path to test JSON file")
    parser.add_argument("--config", default="configs/config.yaml", help="Config file")
    parser.add_argument("--max_samples", type=int, default=None, help="Max samples to evaluate")
    parser.add_argument("--output_file", default=None, help="Save detailed results to JSON")
    parser.add_argument("--verbose", action="store_true", default=True)
    args = parser.parse_args()

    run_full_evaluation(
        adapter_path=args.adapter_path,
        test_file=args.test_file,
        config_path=args.config,
        max_samples=args.max_samples,
        output_file=args.output_file,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
