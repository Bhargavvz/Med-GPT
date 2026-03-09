#!/usr/bin/env python3
"""
MedGPT Visualization Script
Generates training curves, confusion matrix, accuracy charts, and sample heatmaps.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))


def plot_training_curves(trainer_state_path: str, output_dir: str):
    """Plot loss and learning rate curves from trainer_state.json."""
    import matplotlib.pyplot as plt

    with open(trainer_state_path) as f:
        state = json.load(f)

    logs = state.get("log_history", [])

    # Separate train and eval logs
    train_steps, train_loss = [], []
    eval_steps, eval_loss = [], []
    lr_steps, lr_values = [], []

    for entry in logs:
        step = entry.get("step", 0)
        if "loss" in entry:
            train_steps.append(step)
            train_loss.append(entry["loss"])
        if "eval_loss" in entry:
            eval_steps.append(step)
            eval_loss.append(entry["eval_loss"])
        if "learning_rate" in entry:
            lr_steps.append(step)
            lr_values.append(entry["learning_rate"])

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Training Loss
    axes[0].plot(train_steps, train_loss, color="#2196F3", linewidth=1.5, alpha=0.7)
    # Smoothed line
    if len(train_loss) > 20:
        window = max(1, len(train_loss) // 20)
        smoothed = np.convolve(train_loss, np.ones(window)/window, mode='valid')
        axes[0].plot(train_steps[window-1:], smoothed, color="#1565C0", linewidth=2.5, label="Smoothed")
        axes[0].legend()
    axes[0].set_title("Training Loss", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)

    # Eval Loss
    if eval_steps:
        axes[1].plot(eval_steps, eval_loss, "o-", color="#F44336", linewidth=2, markersize=6)
        axes[1].set_title("Validation Loss", fontsize=14, fontweight="bold")
        axes[1].set_xlabel("Step")
        axes[1].set_ylabel("Eval Loss")
        axes[1].grid(True, alpha=0.3)
    else:
        axes[1].text(0.5, 0.5, "No eval data", ha="center", va="center", transform=axes[1].transAxes)

    # Learning Rate
    if lr_steps:
        axes[2].plot(lr_steps, lr_values, color="#4CAF50", linewidth=2)
        axes[2].set_title("Learning Rate Schedule", fontsize=14, fontweight="bold")
        axes[2].set_xlabel("Step")
        axes[2].set_ylabel("LR")
        axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "training_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Training curves: {path}")


def plot_accuracy_breakdown(eval_results_path: str, output_dir: str):
    """Plot accuracy breakdown by dataset and question type."""
    import matplotlib.pyplot as plt

    with open(eval_results_path) as f:
        data = json.load(f)

    metrics = data.get("metrics", data)
    predictions = data.get("predictions", [])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 1. Question type accuracy
    type_data = {}
    if "accuracy_closed" in metrics:
        type_data["Yes/No (Closed)"] = metrics["accuracy_closed"]
    if "accuracy_open" in metrics:
        type_data["Open-ended"] = metrics["accuracy_open"]
    if "accuracy" in metrics:
        type_data["Overall"] = metrics["accuracy"]

    if type_data:
        colors = ["#4CAF50", "#2196F3", "#FF9800"]
        bars = axes[0].bar(type_data.keys(), [v * 100 for v in type_data.values()],
                          color=colors[:len(type_data)], edgecolor="white", linewidth=2)
        for bar, val in zip(bars, type_data.values()):
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        f"{val*100:.1f}%", ha="center", fontweight="bold", fontsize=12)
        axes[0].set_title("Accuracy by Question Type", fontsize=14, fontweight="bold")
        axes[0].set_ylabel("Accuracy (%)")
        axes[0].set_ylim(0, 100)
        axes[0].grid(axis="y", alpha=0.3)

    # 2. Per-dataset accuracy
    dataset_accs = {}
    for key, val in metrics.items():
        if "/" in key and key.endswith("/accuracy"):
            ds_name = key.split("/")[0]
            n_key = f"{ds_name}/n_samples"
            dataset_accs[ds_name] = (val, metrics.get(n_key, 0))

    if dataset_accs:
        names = list(dataset_accs.keys())
        accs = [v[0] * 100 for v in dataset_accs.values()]
        counts = [v[1] for v in dataset_accs.values()]
        colors_ds = plt.cm.Set2(np.linspace(0, 1, len(names)))

        bars = axes[1].bar(names, accs, color=colors_ds, edgecolor="white", linewidth=2)
        for bar, acc, n in zip(bars, accs, counts):
            axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        f"{acc:.1f}%\n(n={int(n)})", ha="center", fontsize=10)
        axes[1].set_title("Accuracy by Dataset", fontsize=14, fontweight="bold")
        axes[1].set_ylabel("Accuracy (%)")
        axes[1].set_ylim(0, 100)
        axes[1].grid(axis="y", alpha=0.3)
        axes[1].tick_params(axis='x', rotation=15)

    plt.tight_layout()
    path = os.path.join(output_dir, "accuracy_breakdown.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Accuracy breakdown: {path}")


def plot_confusion_matrix(eval_results_path: str, output_dir: str):
    """Plot confusion matrix for closed (yes/no) questions."""
    import matplotlib.pyplot as plt

    with open(eval_results_path) as f:
        data = json.load(f)

    predictions = data.get("predictions", [])
    if not predictions:
        print("  ⚠ No predictions found for confusion matrix")
        return

    # Filter closed questions
    closed = [p for p in predictions if p.get("question_type") == "closed"]
    if not closed:
        print("  ⚠ No closed questions found for confusion matrix")
        return

    # Build confusion matrix
    labels = ["yes", "no"]
    matrix = np.zeros((2, 2), dtype=int)
    for p in closed:
        pred_norm = p["prediction_normalized"]
        ref_norm = p["reference_normalized"]
        if ref_norm in labels and pred_norm in labels:
            matrix[labels.index(ref_norm)][labels.index(pred_norm)] += 1

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, cmap="Blues")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred: Yes", "Pred: No"], fontsize=12)
    ax.set_yticklabels(["True: Yes", "True: No"], fontsize=12)

    for i in range(2):
        for j in range(2):
            color = "white" if matrix[i, j] > matrix.max() / 2 else "black"
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center",
                   fontsize=20, fontweight="bold", color=color)

    ax.set_title("Confusion Matrix (Yes/No Questions)", fontsize=14, fontweight="bold")
    plt.colorbar(im)
    plt.tight_layout()
    path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Confusion matrix: {path}")


def plot_metrics_radar(eval_results_path: str, output_dir: str):
    """Plot radar chart of all metrics."""
    import matplotlib.pyplot as plt

    with open(eval_results_path) as f:
        data = json.load(f)

    metrics = data.get("metrics", data)
    categories = ["Accuracy", "BLEU-1", "ROUGE-L", "Token F1"]
    values = [
        metrics.get("accuracy", 0),
        metrics.get("bleu1", 0),
        metrics.get("rouge_l", 0),
        metrics.get("token_f1", 0),
    ]

    # Radar chart
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    values_plot = values + [values[0]]
    angles += [angles[0]]
    categories_plot = categories + [categories[0]]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    ax.fill(angles, values_plot, alpha=0.25, color="#2196F3")
    ax.plot(angles, values_plot, "o-", linewidth=2, color="#1565C0", markersize=8)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=13, fontweight="bold")
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["20%", "40%", "60%", "80%", "100%"], fontsize=9)

    # Add value labels
    for angle, val, cat in zip(angles[:-1], values, categories):
        ax.annotate(f"{val*100:.1f}%", xy=(angle, val), fontsize=11,
                   fontweight="bold", ha="center", va="bottom",
                   xytext=(0, 10), textcoords="offset points")

    ax.set_title("MedGPT Performance Metrics", fontsize=16, fontweight="bold", pad=20)

    plt.tight_layout()
    path = os.path.join(output_dir, "metrics_radar.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Metrics radar: {path}")


def generate_sample_heatmaps(adapter_path: str, test_file: str, output_dir: str,
                              config_path: str = "configs/config.yaml", n_samples: int = 5):
    """Generate Grad-CAM heatmaps for sample predictions."""
    import matplotlib.pyplot as plt
    from models.medgpt import MedGPT, load_config

    config = load_config(config_path)
    print(f"\n  Loading model for heatmaps...")
    medgpt = MedGPT.from_adapter(adapter_path, config=config)

    with open(test_file) as f:
        test_data = json.load(f)

    # Pick samples with existing images
    samples = []
    for s in test_data:
        if os.path.exists(s["image_path"]):
            samples.append(s)
        if len(samples) >= n_samples:
            break

    heatmap_dir = os.path.join(output_dir, "heatmaps")
    os.makedirs(heatmap_dir, exist_ok=True)

    for i, sample in enumerate(samples):
        try:
            pred = medgpt.generate(
                image_path=sample["image_path"],
                question=sample["question"],
                max_new_tokens=64,
                do_sample=False,
            )

            # Try to generate heatmap
            try:
                from models.explainability import GradCAMExplainer
                explainer = GradCAMExplainer(medgpt.model, medgpt.processor)
                heatmap = explainer.generate_heatmap(
                    image_path=sample["image_path"],
                    question=sample["question"],
                )
                if heatmap is not None:
                    from PIL import Image
                    img = Image.open(sample["image_path"]).convert("RGB")
                    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

                    axes[0].imshow(img)
                    axes[0].set_title("Original", fontsize=12)
                    axes[0].axis("off")

                    axes[1].imshow(heatmap, cmap="jet")
                    axes[1].set_title("Attention Heatmap", fontsize=12)
                    axes[1].axis("off")

                    axes[2].imshow(img)
                    axes[2].imshow(heatmap, cmap="jet", alpha=0.4)
                    axes[2].set_title("Overlay", fontsize=12)
                    axes[2].axis("off")

                    fig.suptitle(
                        f"Q: {sample['question'][:60]}\n"
                        f"Pred: {pred[:40]} | Ref: {sample['answer'][:40]}",
                        fontsize=11, fontweight="bold"
                    )
                    plt.tight_layout()
                    path = os.path.join(heatmap_dir, f"heatmap_{i}.png")
                    plt.savefig(path, dpi=150, bbox_inches="tight")
                    plt.close()
                    print(f"  ✓ Heatmap {i}: {path}")
                    continue
            except Exception as e:
                pass

            # Fallback: just show prediction on image
            from PIL import Image
            img = Image.open(sample["image_path"]).convert("RGB")
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.imshow(img)
            ax.set_title(
                f"Q: {sample['question'][:60]}\n"
                f"Pred: {pred[:50]} | Ref: {sample['answer'][:50]}",
                fontsize=10, fontweight="bold"
            )
            ax.axis("off")
            plt.tight_layout()
            path = os.path.join(heatmap_dir, f"sample_{i}.png")
            plt.savefig(path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"  ✓ Sample {i}: {path}")

        except Exception as e:
            print(f"  ✗ Sample {i} failed: {e}")


def main():
    parser = argparse.ArgumentParser(description="MedGPT Visualizations")
    parser.add_argument("--output_dir", default="results/figures", help="Output directory for figures")
    parser.add_argument("--eval_results", default="results/eval_results.json", help="Evaluation results JSON")
    parser.add_argument("--trainer_state", default=None, help="Path to trainer_state.json")
    parser.add_argument("--adapter_path", default="checkpoints/finetune/best_model", help="Adapter path for heatmaps")
    parser.add_argument("--test_file", default="data/processed/finetune_test.json", help="Test file for heatmaps")
    parser.add_argument("--config", default="configs/config.yaml", help="Config file")
    parser.add_argument("--skip_heatmaps", action="store_true", help="Skip heatmap generation (faster)")
    parser.add_argument("--n_heatmaps", type=int, default=5, help="Number of heatmap samples")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"\nGenerating visualizations in: {args.output_dir}")

    # Auto-find trainer_state.json
    trainer_state = args.trainer_state
    if not trainer_state:
        for ckpt in ["checkpoint-2472", "checkpoint-2000", "checkpoint-1500"]:
            path = f"checkpoints/finetune/{ckpt}/trainer_state.json"
            if os.path.exists(path):
                trainer_state = path
                break

    # 1. Training curves
    if trainer_state and os.path.exists(trainer_state):
        print("\n[1/4] Training Curves")
        plot_training_curves(trainer_state, args.output_dir)
    else:
        print("\n[1/4] Skipping training curves (no trainer_state.json found)")

    # 2. Accuracy breakdown
    if os.path.exists(args.eval_results):
        print("\n[2/4] Accuracy Breakdown")
        plot_accuracy_breakdown(args.eval_results, args.output_dir)

        # 3. Confusion matrix
        print("\n[3/4] Confusion Matrix")
        plot_confusion_matrix(args.eval_results, args.output_dir)

        # 4. Metrics radar
        print("\n[3.5/4] Metrics Radar")
        plot_metrics_radar(args.eval_results, args.output_dir)
    else:
        print(f"\n[2-3/4] Skipping (no eval results at {args.eval_results})")
        print("  Run evaluation first: python training/evaluate.py ...")

    # 4. Sample heatmaps
    if not args.skip_heatmaps:
        print("\n[4/4] Sample Heatmaps")
        generate_sample_heatmaps(
            adapter_path=args.adapter_path,
            test_file=args.test_file,
            output_dir=args.output_dir,
            config_path=args.config,
            n_samples=args.n_heatmaps,
        )
    else:
        print("\n[4/4] Skipping heatmaps (--skip_heatmaps)")

    print(f"\n{'='*60}")
    print(f"All visualizations saved to: {args.output_dir}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
