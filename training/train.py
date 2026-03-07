#!/usr/bin/env python3
"""
MedGPT Training Pipeline
Two-stage training: PMC-VQA pre-training → VQA-RAD+SLAKE+PathVQA fine-tuning.
Uses HuggingFace Trainer with LoRA on Qwen3-VL-8B-Instruct.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from transformers import (
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from data.dataset import MedicalVQADataset, collate_fn
from models.medgpt import MedGPT, load_config
from training.evaluate import normalize_answer, compute_metrics_from_predictions


class MedicalVQATrainer(Trainer):
    """Custom Trainer that strips metadata keys before forward pass."""

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Forward pass with metadata stripping."""
        # Remove metadata keys that aren't model inputs
        model_inputs = {
            k: v for k, v in inputs.items()
            if not k.startswith("_") and isinstance(v, torch.Tensor)
        }
        outputs = model(**model_inputs)
        loss = outputs.loss

        if return_outputs:
            return loss, outputs
        return loss


def run_evaluation(medgpt, dataset, max_samples=100, batch_size=1):
    """Run evaluation by generating predictions and computing metrics."""
    model = medgpt.model
    processor = medgpt.processor
    model.eval()

    predictions = []
    references = []
    question_types = []

    n_samples = min(len(dataset), max_samples)
    print(f"\n  Evaluating on {n_samples} samples...")

    for i in range(n_samples):
        sample = dataset.samples[i]
        try:
            pred = medgpt.generate(
                image_path=sample["image_path"],
                question=sample["question"],
                knowledge=sample.get("knowledge", ""),
                max_new_tokens=64,
                do_sample=False,
            )
            predictions.append(pred)
            references.append(sample["answer"])
            question_types.append(sample.get("question_type", "open"))

            if i < 5:
                print(f"    [{i}] Q: {sample['question'][:60]}")
                print(f"         Pred: {pred[:60]}")
                print(f"         Ref:  {sample['answer'][:60]}")
        except Exception as e:
            print(f"    [{i}] Error: {e}")
            predictions.append("")
            references.append(sample["answer"])
            question_types.append(sample.get("question_type", "open"))

    # Compute metrics
    metrics = compute_metrics_from_predictions(predictions, references, question_types)
    return metrics


def train_stage(
    config: dict,
    stage: str,
    medgpt: MedGPT,
    resume_from: str = None,
    dry_run: bool = False,
    max_steps: int = -1,
) -> str:
    """
    Run one stage of training.

    Args:
        config: Configuration dict
        stage: "pretrain" or "finetune"
        medgpt: MedGPT model instance
        resume_from: Path to checkpoint to resume from
        dry_run: If True, run only 2 steps
        max_steps: Maximum training steps (-1 for full training)

    Returns:
        Path to the output directory with the trained adapter
    """
    stage_config = config["training"][stage]
    print(f"\n{'=' * 60}")
    print(f"STAGE: {stage.upper()}")
    print(f"{'=' * 60}")

    # Load datasets
    train_file = stage_config["data_file"]
    val_file = stage_config["val_file"]

    if not os.path.exists(train_file):
        print(f"ERROR: Training data not found: {train_file}")
        print("Run data/prepare_data.py first!")
        sys.exit(1)

    train_dataset = MedicalVQADataset(
        data_file=train_file,
        processor=medgpt.processor,
        max_seq_length=stage_config.get("max_seq_length", 1024),
        is_training=True,
    )
    val_dataset = MedicalVQADataset(
        data_file=val_file,
        processor=medgpt.processor,
        max_seq_length=stage_config.get("max_seq_length", 1024),
        is_training=True,
    )

    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples: {len(val_dataset)}")

    # Dry run overrides
    if dry_run:
        max_steps = 2
        stage_config["eval_steps"] = 2
        stage_config["save_steps"] = 2
        stage_config["logging_steps"] = 1

    # Training arguments
    output_dir = stage_config["output_dir"]
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=stage_config["num_epochs"],
        per_device_train_batch_size=stage_config["batch_size"],
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=stage_config["gradient_accumulation_steps"],
        learning_rate=stage_config["learning_rate"],
        lr_scheduler_type=stage_config.get("lr_scheduler", "cosine"),
        warmup_ratio=stage_config.get("warmup_ratio", 0.05),
        max_grad_norm=stage_config.get("max_grad_norm", 1.0),
        bf16=config["training"].get("bf16", True),
        fp16=config["training"].get("fp16", False),
        eval_strategy="steps",
        eval_steps=stage_config.get("eval_steps", 100),
        save_strategy="steps",
        save_steps=stage_config.get("save_steps", 100),
        save_total_limit=3,
        logging_steps=stage_config.get("logging_steps", 10),
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=config["training"].get("report_to", "none"),
        seed=config["training"].get("seed", 42),
        dataloader_num_workers=config["training"].get("dataloader_num_workers", 4),
        remove_unused_columns=False,
        max_steps=max_steps if max_steps > 0 else -1,
        dataloader_pin_memory=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    # Create trainer
    trainer = MedicalVQATrainer(
        model=medgpt.model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn,
    )

    # Train
    print(f"\n  Starting training...")
    if resume_from:
        print(f"  Resuming from: {resume_from}")
        trainer.train(resume_from_checkpoint=resume_from)
    else:
        trainer.train()

    # Save best model
    best_model_dir = os.path.join(output_dir, "best_model")
    medgpt.save_adapter(best_model_dir)
    print(f"\n  Best model saved to: {best_model_dir}")

    # Run evaluation on val set
    if not dry_run:
        print("\n  Running evaluation on validation set...")
        metrics = run_evaluation(medgpt, val_dataset, max_samples=200)
        print(f"\n  Validation Results:")
        for k, v in metrics.items():
            print(f"    {k}: {v:.4f}")

        # Save metrics
        metrics_file = os.path.join(output_dir, "eval_results.json")
        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=2)

    return best_model_dir


def main():
    parser = argparse.ArgumentParser(description="MedGPT Training Pipeline")
    parser.add_argument("--config", default="configs/config.yaml", help="Config file path")
    parser.add_argument("--stage", choices=["pretrain", "finetune", "both"],
                        default="both", help="Training stage to run")
    parser.add_argument("--data_dir", default=None, help="Override data directory")
    parser.add_argument("--output_dir", default=None, help="Override output directory")
    parser.add_argument("--num_epochs", type=int, default=None, help="Override num epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--eval_steps", type=int, default=None)
    parser.add_argument("--max_steps", type=int, default=-1, help="Max training steps")
    parser.add_argument("--resume_from", default=None, help="Resume from checkpoint")
    parser.add_argument("--dry_run", action="store_true", help="Quick test run (2 steps)")
    parser.add_argument("--eval_only", action="store_true", help="Only run evaluation")
    parser.add_argument("--adapter_path", default=None, help="Adapter path for eval_only")
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Apply overrides
    for stage_name in ["pretrain", "finetune"]:
        stage_cfg = config["training"][stage_name]
        if args.num_epochs is not None:
            stage_cfg["num_epochs"] = args.num_epochs
        if args.batch_size is not None:
            stage_cfg["batch_size"] = args.batch_size
        if args.gradient_accumulation_steps is not None:
            stage_cfg["gradient_accumulation_steps"] = args.gradient_accumulation_steps
        if args.learning_rate is not None:
            stage_cfg["learning_rate"] = args.learning_rate
        if args.eval_steps is not None:
            stage_cfg["eval_steps"] = args.eval_steps
        if args.output_dir is not None:
            stage_cfg["output_dir"] = os.path.join(args.output_dir, stage_name)
        if args.data_dir is not None:
            stage_cfg["data_file"] = os.path.join(args.data_dir, f"{stage_name}_train.json")
            stage_cfg["val_file"] = os.path.join(args.data_dir, f"{stage_name}_val.json")

    # Eval-only mode
    if args.eval_only:
        adapter = args.adapter_path or config["inference"]["adapter_path"]
        print(f"Loading model from adapter: {adapter}")
        medgpt = MedGPT.from_adapter(adapter, config=config)
        test_file = os.path.join(
            config["data"]["processed_dir"], "finetune_test.json"
        )
        test_dataset = MedicalVQADataset(
            data_file=test_file,
            processor=medgpt.processor,
            is_training=False,
        )
        metrics = run_evaluation(medgpt, test_dataset, max_samples=len(test_dataset))
        print("\nTest Results:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")
        return

    # Initialize model
    print("Initializing MedGPT...")
    medgpt = MedGPT(config=config, training=True)

    # Stage 1: Pre-training
    pretrain_adapter = None
    if args.stage in ("pretrain", "both"):
        if config["training"]["pretrain"]["enabled"]:
            pretrain_adapter = train_stage(
                config, "pretrain", medgpt,
                resume_from=args.resume_from,
                dry_run=args.dry_run,
                max_steps=args.max_steps,
            )

    # Stage 2: Fine-tuning
    if args.stage in ("finetune", "both"):
        if config["training"]["finetune"]["enabled"]:
            # If we just pre-trained, the LoRA weights are already loaded
            # If starting fresh fine-tune, reinitialize LoRA (or load pretrained adapter)
            if pretrain_adapter and args.stage == "both":
                print("\nLoading pre-trained adapter for fine-tuning...")
                # The model already has the pre-trained weights from stage 1

            train_stage(
                config, "finetune", medgpt,
                resume_from=args.resume_from if args.stage == "finetune" else None,
                dry_run=args.dry_run,
                max_steps=args.max_steps,
            )

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
