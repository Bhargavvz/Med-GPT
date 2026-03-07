#!/usr/bin/env python3
"""
MedGPT Inference Pipeline
Load a trained model and run inference on individual images.
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    parser = argparse.ArgumentParser(description="MedGPT Inference")
    parser.add_argument("--image", required=True, help="Path to medical image")
    parser.add_argument("--question", required=True, help="Question about the image")
    parser.add_argument("--knowledge", default="", help="Optional medical context")
    parser.add_argument("--adapter_path", default=None, help="Path to LoRA adapter")
    parser.add_argument("--config", default="configs/config.yaml", help="Config file")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--gradcam", action="store_true", help="Generate Grad-CAM heatmap")
    parser.add_argument("--gradcam_output", default="gradcam_output.png", help="Heatmap output path")
    parser.add_argument("--batch", default=None, help="JSON file with multiple queries")
    parser.add_argument("--output", default=None, help="Output file for batch results")
    args = parser.parse_args()

    from models.medgpt import MedGPT, load_config

    config = load_config(args.config)
    adapter_path = args.adapter_path or config["inference"]["adapter_path"]

    print(f"Loading model from: {adapter_path}")
    medgpt = MedGPT.from_adapter(adapter_path, config=config)

    if args.batch:
        # Batch inference
        with open(args.batch) as f:
            queries = json.load(f)

        results = []
        for i, q in enumerate(queries):
            print(f"\n[{i+1}/{len(queries)}] Processing...")
            answer = medgpt.generate(
                image_path=q["image"],
                question=q["question"],
                knowledge=q.get("knowledge", ""),
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                do_sample=args.do_sample,
            )
            result = {
                "image": q["image"],
                "question": q["question"],
                "answer": answer,
            }
            results.append(result)
            print(f"  Q: {q['question']}")
            print(f"  A: {answer}")

        if args.output:
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to: {args.output}")
    else:
        # Single inference
        if not os.path.exists(args.image):
            print(f"Error: Image not found: {args.image}")
            sys.exit(1)

        print(f"\nImage: {args.image}")
        print(f"Question: {args.question}")
        if args.knowledge:
            print(f"Knowledge: {args.knowledge}")

        answer = medgpt.generate(
            image_path=args.image,
            question=args.question,
            knowledge=args.knowledge,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            do_sample=args.do_sample,
        )

        print(f"\nAnswer: {answer}")

        # Grad-CAM
        if args.gradcam:
            print(f"\nGenerating Grad-CAM heatmap...")
            from models.explainability import GradCAM

            gradcam = GradCAM(medgpt.model, medgpt.processor)
            gradcam.save_heatmap(
                image_path=args.image,
                question=args.question,
                output_path=args.gradcam_output,
                knowledge=args.knowledge,
            )


if __name__ == "__main__":
    main()
