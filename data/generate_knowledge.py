#!/usr/bin/env python3
"""
MedGPT Knowledge Generator
Generate medical knowledge snippets for training samples using an LLM API.
This is optional — it enhances accuracy by +5-10% via RAG-style prompt injection.
"""

import argparse
import json
import os
import time
from pathlib import Path


def generate_knowledge_prompt(question: str, answer: str) -> str:
    """Create a prompt for the LLM to generate relevant medical context."""
    return f"""Given this medical visual question answering pair, generate a brief (2-3 sentences) 
medical knowledge snippet that would help a model answer this question correctly. 
The snippet should contain relevant medical facts, terminology, and context.

Question: {question}
Answer: {answer}

Generate only the knowledge snippet, nothing else. Focus on factual medical information 
that explains WHY this answer is correct."""


def generate_with_openai(prompt: str, api_key: str, model: str = "gpt-4o-mini") -> str:
    """Generate knowledge using OpenAI API."""
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a medical knowledge expert. Provide concise, factual medical information."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=150,
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"  [ERROR] OpenAI API error: {e}")
        return ""


def generate_with_anthropic(prompt: str, api_key: str, model: str = "claude-3-5-haiku-20241022") -> str:
    """Generate knowledge using Anthropic API."""
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model=model,
            max_tokens=150,
            messages=[
                {"role": "user", "content": prompt}
            ],
            system="You are a medical knowledge expert. Provide concise, factual medical information.",
        )
        return response.content[0].text.strip()
    except Exception as e:
        print(f"  [ERROR] Anthropic API error: {e}")
        return ""


def main():
    parser = argparse.ArgumentParser(description="Generate medical knowledge snippets")
    parser.add_argument("--input_file", required=True, help="Input JSON data file")
    parser.add_argument("--output_file", required=True, help="Output JSON with knowledge added")
    parser.add_argument("--provider", choices=["openai", "anthropic"], default="openai")
    parser.add_argument("--api_key", default=None, help="API key (or set OPENAI_API_KEY / ANTHROPIC_API_KEY env var)")
    parser.add_argument("--model", default=None, help="Model to use")
    parser.add_argument("--batch_size", type=int, default=50, help="Save progress every N samples")
    parser.add_argument("--max_samples", type=int, default=None, help="Max samples to process")
    parser.add_argument("--delay", type=float, default=0.1, help="Delay between API calls (seconds)")
    args = parser.parse_args()

    # Get API key
    api_key = args.api_key
    if not api_key:
        env_var = "OPENAI_API_KEY" if args.provider == "openai" else "ANTHROPIC_API_KEY"
        api_key = os.environ.get(env_var)
    if not api_key:
        print(f"Error: No API key provided. Set --api_key or {env_var} env var.")
        return

    # Set default model
    if not args.model:
        args.model = "gpt-4o-mini" if args.provider == "openai" else "claude-3-5-haiku-20241022"

    # Load data
    with open(args.input_file) as f:
        samples = json.load(f)

    # Load existing progress if output file exists
    processed = {}
    if os.path.exists(args.output_file):
        with open(args.output_file) as f:
            existing = json.load(f)
        for s in existing:
            if s.get("knowledge"):
                key = f"{s['question']}||{s['answer']}"
                processed[key] = s["knowledge"]
        print(f"Loaded {len(processed)} existing knowledge snippets")

    if args.max_samples:
        samples = samples[:args.max_samples]

    generate_fn = generate_with_openai if args.provider == "openai" else generate_with_anthropic

    print(f"Processing {len(samples)} samples with {args.provider}/{args.model}")

    for i, sample in enumerate(samples):
        key = f"{sample['question']}||{sample['answer']}"
        if key in processed:
            sample["knowledge"] = processed[key]
            continue

        prompt = generate_knowledge_prompt(sample["question"], sample["answer"])
        knowledge = generate_fn(prompt, api_key, args.model)
        sample["knowledge"] = knowledge

        if knowledge:
            processed[key] = knowledge

        if (i + 1) % args.batch_size == 0:
            # Save progress
            with open(args.output_file, "w") as f:
                json.dump(samples, f, indent=2)
            print(f"  Progress: {i + 1}/{len(samples)} ({len(processed)} with knowledge)")

        if args.delay > 0:
            time.sleep(args.delay)

    # Final save
    with open(args.output_file, "w") as f:
        json.dump(samples, f, indent=2)

    knowledge_count = sum(1 for s in samples if s.get("knowledge"))
    print(f"\nDone! {knowledge_count}/{len(samples)} samples have knowledge snippets")
    print(f"Output saved to: {args.output_file}")


if __name__ == "__main__":
    main()
