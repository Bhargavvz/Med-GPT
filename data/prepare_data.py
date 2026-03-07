#!/usr/bin/env python3
"""
MedGPT Data Preparation Pipeline
Downloads VQA-RAD, SLAKE, PathVQA, and PMC-VQA from HuggingFace,
converts to unified JSON format, and creates train/val/test splits.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from io import BytesIO
from collections import Counter

from datasets import load_dataset
from PIL import Image
from tqdm import tqdm


def save_image(image, save_path: str) -> bool:
    """Save a PIL image to disk. Returns True if successful."""
    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if isinstance(image, Image.Image):
            image = image.convert("RGB")
            image.save(save_path, "JPEG", quality=95)
            return True
        return False
    except Exception as e:
        print(f"  [WARN] Failed to save image {save_path}: {e}")
        return False


def classify_question_type(answer: str) -> str:
    """Classify whether a QA pair is closed (yes/no) or open-ended."""
    if answer is None:
        return "open"
    ans_lower = str(answer).strip().lower()
    closed_answers = {"yes", "no", "true", "false", "y", "n", "1", "0"}
    if ans_lower in closed_answers:
        return "closed"
    return "open"


# =========================================================================
# VQA-RAD
# =========================================================================
def prepare_vqa_rad(output_dir: str, images_dir: str) -> list:
    """Download and prepare VQA-RAD dataset."""
    print("\n" + "=" * 60)
    print("Preparing VQA-RAD (flaviagiammarino/vqa-rad)")
    print("=" * 60)

    ds = load_dataset("flaviagiammarino/vqa-rad", trust_remote_code=True)
    img_dir = os.path.join(images_dir, "vqa_rad")
    os.makedirs(img_dir, exist_ok=True)

    all_samples = []
    for split_name in ds:
        split = ds[split_name]
        print(f"  Processing split '{split_name}': {len(split)} samples")
        for idx, sample in enumerate(tqdm(split, desc=f"  {split_name}")):
            image = sample.get("image")
            question = sample.get("question", "")
            answer = str(sample.get("answer", ""))

            if not question or not answer:
                continue

            img_filename = f"vqa_rad_{split_name}_{idx:05d}.jpg"
            img_path = os.path.join(img_dir, img_filename)

            if not os.path.exists(img_path):
                if not save_image(image, img_path):
                    continue

            all_samples.append({
                "image_path": os.path.abspath(img_path),
                "question": question.strip(),
                "answer": answer.strip(),
                "question_type": classify_question_type(answer),
                "dataset": "vqa_rad",
                "split": split_name,
                "knowledge": ""
            })

    print(f"  Total VQA-RAD samples: {len(all_samples)}")
    return all_samples


# =========================================================================
# SLAKE
# =========================================================================
def prepare_slake(output_dir: str, images_dir: str) -> list:
    """Download and prepare SLAKE dataset (English only)."""
    print("\n" + "=" * 60)
    print("Preparing SLAKE (BoKelworker/SLAKE)")
    print("=" * 60)

    ds = load_dataset("BoKelworker/SLAKE", trust_remote_code=True)
    img_dir = os.path.join(images_dir, "slake")
    os.makedirs(img_dir, exist_ok=True)

    all_samples = []
    for split_name in ds:
        split = ds[split_name]
        print(f"  Processing split '{split_name}': {len(split)} samples")
        for idx, sample in enumerate(tqdm(split, desc=f"  {split_name}")):
            # SLAKE has a 'q_lang' field — only keep English
            q_lang = sample.get("q_lang", "en")
            if q_lang != "en":
                continue

            image = sample.get("image")
            question = sample.get("question", "")
            answer = str(sample.get("answer", ""))

            if not question or not answer:
                continue

            img_filename = f"slake_{split_name}_{idx:05d}.jpg"
            img_path = os.path.join(img_dir, img_filename)

            if not os.path.exists(img_path):
                if not save_image(image, img_path):
                    continue

            # SLAKE provides answer_type directly
            q_type = sample.get("answer_type", "")
            if q_type.lower() in ("closed", "yes/no"):
                q_type = "closed"
            else:
                q_type = classify_question_type(answer)

            all_samples.append({
                "image_path": os.path.abspath(img_path),
                "question": question.strip(),
                "answer": answer.strip(),
                "question_type": q_type,
                "dataset": "slake",
                "split": split_name,
                "knowledge": ""
            })

    print(f"  Total SLAKE samples (English): {len(all_samples)}")
    return all_samples


# =========================================================================
# PathVQA
# =========================================================================
def prepare_pathvqa(output_dir: str, images_dir: str) -> list:
    """Download and prepare PathVQA dataset."""
    print("\n" + "=" * 60)
    print("Preparing PathVQA (flaviagiammarino/path-vqa)")
    print("=" * 60)

    ds = load_dataset("flaviagiammarino/path-vqa", trust_remote_code=True)
    img_dir = os.path.join(images_dir, "pathvqa")
    os.makedirs(img_dir, exist_ok=True)

    all_samples = []
    for split_name in ds:
        split = ds[split_name]
        print(f"  Processing split '{split_name}': {len(split)} samples")
        for idx, sample in enumerate(tqdm(split, desc=f"  {split_name}")):
            image = sample.get("image")
            question = sample.get("question", "")
            answer = str(sample.get("answer", ""))

            if not question or not answer:
                continue

            img_filename = f"pathvqa_{split_name}_{idx:05d}.jpg"
            img_path = os.path.join(img_dir, img_filename)

            if not os.path.exists(img_path):
                if not save_image(image, img_path):
                    continue

            all_samples.append({
                "image_path": os.path.abspath(img_path),
                "question": question.strip(),
                "answer": answer.strip(),
                "question_type": classify_question_type(answer),
                "dataset": "pathvqa",
                "split": split_name,
                "knowledge": ""
            })

    print(f"  Total PathVQA samples: {len(all_samples)}")
    return all_samples


# =========================================================================
# PMC-VQA
# =========================================================================
def prepare_pmc_vqa(output_dir: str, images_dir: str) -> list:
    """Download and prepare PMC-VQA dataset."""
    print("\n" + "=" * 60)
    print("Preparing PMC-VQA (xmcmic/PMC-VQA)")
    print("=" * 60)

    ds = load_dataset("xmcmic/PMC-VQA", trust_remote_code=True)
    img_dir = os.path.join(images_dir, "pmc_vqa")
    os.makedirs(img_dir, exist_ok=True)

    all_samples = []
    for split_name in ds:
        split = ds[split_name]
        print(f"  Processing split '{split_name}': {len(split)} samples")
        for idx, sample in enumerate(tqdm(split, desc=f"  {split_name}")):
            image = sample.get("image")
            question = sample.get("Question", sample.get("question", ""))
            answer = str(sample.get("Answer", sample.get("answer", "")))

            if not question or not answer:
                continue

            img_filename = f"pmc_vqa_{split_name}_{idx:06d}.jpg"
            img_path = os.path.join(img_dir, img_filename)

            if not os.path.exists(img_path):
                if not save_image(image, img_path):
                    continue

            # PMC-VQA may have multiple choice — extract the answer
            # Some entries have "Choice": "A", with options
            choice = sample.get("Choice", "")
            if choice and "A" in sample:
                # It's multiple choice: extract the actual answer text
                choice_key = choice.strip().upper()
                answer_text = sample.get(choice_key, answer)
                answer = str(answer_text).strip()

            all_samples.append({
                "image_path": os.path.abspath(img_path),
                "question": question.strip(),
                "answer": answer.strip(),
                "question_type": classify_question_type(answer),
                "dataset": "pmc_vqa",
                "split": split_name,
                "knowledge": ""
            })

    print(f"  Total PMC-VQA samples: {len(all_samples)}")
    return all_samples


# =========================================================================
# Splitting & Saving
# =========================================================================
def create_splits(samples: list, train_ratio=0.8, val_ratio=0.1) -> dict:
    """Create train/val/test splits from samples.
    If samples already have a 'split' field with 'train'/'test'/'validation',
    respect those splits. Otherwise, create splits randomly.
    """
    import random
    random.seed(42)

    # Check if dataset already has splits
    has_splits = any(s.get("split") in ("train", "test", "validation", "val") for s in samples)

    if has_splits:
        train = [s for s in samples if s.get("split") == "train"]
        test = [s for s in samples if s.get("split") == "test"]
        val = [s for s in samples if s.get("split") in ("validation", "val")]

        # If no explicit validation split, carve from train
        if not val and train:
            random.shuffle(train)
            val_size = max(1, int(len(train) * 0.1))
            val = train[:val_size]
            train = train[val_size:]

        # If still no test but we have "train" only, carve test+val
        if not test and train:
            random.shuffle(train)
            test_size = max(1, int(len(train) * 0.1))
            test = train[:test_size]
            train = train[test_size:]
    else:
        random.shuffle(samples)
        n = len(samples)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        train = samples[:train_end]
        val = samples[train_end:val_end]
        test = samples[val_end:]

    # Remove the 'split' key from each sample
    for s in train + val + test:
        s.pop("split", None)

    return {"train": train, "val": val, "test": test}


def save_splits(splits: dict, output_dir: str, prefix: str = ""):
    """Save train/val/test splits as JSON files."""
    os.makedirs(output_dir, exist_ok=True)
    for split_name, samples in splits.items():
        filename = f"{prefix}_{split_name}.json" if prefix else f"{split_name}.json"
        filepath = os.path.join(output_dir, filename)
        with open(filepath, "w") as f:
            json.dump(samples, f, indent=2)
        print(f"  Saved {filepath}: {len(samples)} samples")


def validate_data(samples: list) -> list:
    """Validate that all samples have existing images and non-empty fields."""
    valid = []
    invalid_count = 0
    for s in samples:
        if not os.path.exists(s["image_path"]):
            invalid_count += 1
            continue
        if not s["question"].strip() or not s["answer"].strip():
            invalid_count += 1
            continue
        valid.append(s)

    if invalid_count > 0:
        print(f"  [WARN] Removed {invalid_count} invalid samples")
    return valid


def print_stats(samples: list, name: str = ""):
    """Print dataset statistics."""
    if not samples:
        print(f"  {name}: 0 samples")
        return

    q_types = Counter(s["question_type"] for s in samples)
    datasets = Counter(s["dataset"] for s in samples)

    print(f"\n  {name} Statistics:")
    print(f"    Total samples: {len(samples)}")
    print(f"    Question types: {dict(q_types)}")
    print(f"    Datasets: {dict(datasets)}")


def main():
    parser = argparse.ArgumentParser(description="Prepare Medical VQA datasets")
    parser.add_argument("--datasets", nargs="+",
                        choices=["vqa_rad", "slake", "pathvqa", "pmc_vqa", "all"],
                        default=["all"],
                        help="Which datasets to download and prepare")
    parser.add_argument("--output_dir", type=str, default="data/processed",
                        help="Output directory for processed JSON files")
    parser.add_argument("--images_dir", type=str, default="data/images",
                        help="Directory to save downloaded images")
    parser.add_argument("--validate", action="store_true",
                        help="Validate all images exist and are loadable")
    parser.add_argument("--skip_existing", action="store_true", default=True,
                        help="Skip datasets whose output files already exist")
    args = parser.parse_args()

    if "all" in args.datasets:
        args.datasets = ["vqa_rad", "slake", "pathvqa", "pmc_vqa"]

    # Make paths absolute
    args.output_dir = os.path.abspath(args.output_dir)
    args.images_dir = os.path.abspath(args.images_dir)

    print(f"Output directory: {args.output_dir}")
    print(f"Images directory: {args.images_dir}")
    print(f"Datasets to prepare: {args.datasets}")

    # Prepare datasets for each stage
    pretrain_samples = []    # PMC-VQA → stage 1
    finetune_samples = []    # VQA-RAD + SLAKE + PathVQA → stage 2

    preparers = {
        "vqa_rad": prepare_vqa_rad,
        "slake": prepare_slake,
        "pathvqa": prepare_pathvqa,
        "pmc_vqa": prepare_pmc_vqa,
    }

    pretrain_datasets = {"pmc_vqa"}
    finetune_datasets = {"vqa_rad", "slake", "pathvqa"}

    for ds_name in args.datasets:
        if ds_name not in preparers:
            print(f"Unknown dataset: {ds_name}")
            continue

        samples = preparers[ds_name](args.output_dir, args.images_dir)

        if args.validate:
            samples = validate_data(samples)

        if ds_name in pretrain_datasets:
            pretrain_samples.extend(samples)
        else:
            finetune_samples.extend(samples)

    # Create and save splits
    if pretrain_samples:
        print("\n" + "=" * 60)
        print("Creating PRE-TRAINING splits (PMC-VQA)")
        print("=" * 60)
        pretrain_splits = create_splits(pretrain_samples)
        save_splits(pretrain_splits, args.output_dir, prefix="pretrain")
        for split_name, split_data in pretrain_splits.items():
            print_stats(split_data, f"pretrain_{split_name}")

    if finetune_samples:
        print("\n" + "=" * 60)
        print("Creating FINE-TUNING splits (VQA-RAD + SLAKE + PathVQA)")
        print("=" * 60)
        finetune_splits = create_splits(finetune_samples)
        save_splits(finetune_splits, args.output_dir, prefix="finetune")
        for split_name, split_data in finetune_splits.items():
            print_stats(split_data, f"finetune_{split_name}")

    print("\n" + "=" * 60)
    print("DATA PREPARATION COMPLETE")
    print("=" * 60)
    print(f"Pre-training samples: {len(pretrain_samples)}")
    print(f"Fine-tuning samples: {len(finetune_samples)}")
    print(f"Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()
