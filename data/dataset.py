#!/usr/bin/env python3
"""
MedGPT Dataset Class
PyTorch Dataset for Medical VQA with proper chat templates,
label masking, and variable-length collation using Qwen3-VL processor.
"""

import json
import os
from typing import Dict, List, Optional

import torch
from PIL import Image
from torch.utils.data import Dataset


class MedicalVQADataset(Dataset):
    """
    Dataset for Medical VQA that uses Qwen3-VL's native processor.

    Key design decisions:
    1. Uses the model's own processor — no separate tokenizer
    2. Builds proper chat-template messages with image tokens
    3. Masks all prompt tokens with -100 — only answer tokens contribute to loss
    4. Handles missing/corrupt images gracefully
    """

    # System prompt for medical VQA
    SYSTEM_PROMPT = (
        "You are an expert medical imaging specialist with extensive training in "
        "radiology, pathology, and clinical diagnosis. Analyze the provided medical "
        "image carefully and answer the question accurately and concisely. "
        "Provide a brief medical rationale for your answer when appropriate."
    )

    def __init__(
        self,
        data_file: str,
        processor=None,
        max_seq_length: int = 1024,
        is_training: bool = True,
        include_rationale: bool = True,
    ):
        """
        Args:
            data_file: Path to JSON file with samples
            processor: Qwen3-VL AutoProcessor instance
            max_seq_length: Maximum sequence length for tokenization
            is_training: If True, include answers in the prompt for training
            include_rationale: If True, prompt for rationale in answers
        """
        self.data_file = data_file
        self.processor = processor
        self.max_seq_length = max_seq_length
        self.is_training = is_training
        self.include_rationale = include_rationale

        # Load data
        with open(data_file, "r") as f:
            self.samples = json.load(f)

        print(f"Loaded {len(self.samples)} samples from {data_file}")

        # Filter out samples with missing images
        valid_samples = []
        missing_count = 0
        for s in self.samples:
            if os.path.exists(s["image_path"]):
                valid_samples.append(s)
            else:
                missing_count += 1
        if missing_count > 0:
            print(f"  [WARN] Skipped {missing_count} samples with missing images")
        self.samples = valid_samples

    def __len__(self) -> int:
        return len(self.samples)

    def _build_messages(self, sample: dict) -> tuple:
        """Build chat messages for Qwen3-VL.

        Returns:
            (messages_with_answer, answer_text) for training
            (messages_without_answer, None) for inference
        """
        # Build user content as a list of content items
        user_content = []

        # 1. Add image
        user_content.append({
            "type": "image",
            "image": sample["image_path"],
        })

        # 2. Build text content
        text_parts = []

        # Add knowledge context if available
        knowledge = sample.get("knowledge", "")
        if knowledge:
            text_parts.append(f"Medical Context: {knowledge}")

        # Add the question
        text_parts.append(f"Question: {sample['question']}")

        if self.include_rationale and self.is_training:
            text_parts.append(
                "Provide your answer followed by a brief medical rationale."
            )

        user_content.append({
            "type": "text",
            "text": "\n".join(text_parts),
        })

        # Build messages
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        if self.is_training:
            answer_text = sample["answer"]
            messages.append({
                "role": "assistant",
                "content": answer_text,
            })
            return messages, answer_text
        else:
            return messages, None

    def __getitem__(self, idx: int) -> dict:
        """Get a single sample, tokenized and ready for the model."""
        sample = self.samples[idx]
        messages, answer_text = self._build_messages(sample)

        # Use the processor to create model inputs
        # For training, we need to tokenize the full conversation and mask prompt tokens
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

        # Load the image
        try:
            image = Image.open(sample["image_path"]).convert("RGB")
        except Exception as e:
            print(f"[WARN] Failed to load image {sample['image_path']}: {e}")
            # Return a small black image as fallback
            image = Image.new("RGB", (224, 224), (0, 0, 0))

        # Process with the Qwen3-VL processor
        inputs = self.processor(
            text=[text],
            images=[image],
            padding=False,
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors="pt",
        )

        # Squeeze batch dimension
        input_ids = inputs["input_ids"].squeeze(0)
        attention_mask = inputs["attention_mask"].squeeze(0)

        # Extract pixel_values and image_grid_thw if present
        pixel_values = inputs.get("pixel_values", None)
        if pixel_values is not None:
            pixel_values = pixel_values.squeeze(0) if pixel_values.dim() > 3 else pixel_values
        image_grid_thw = inputs.get("image_grid_thw", None)
        if image_grid_thw is not None:
            image_grid_thw = image_grid_thw.squeeze(0) if image_grid_thw.dim() > 1 else image_grid_thw

        # Create labels with prompt masking
        labels = input_ids.clone()

        if self.is_training and answer_text:
            # Tokenize just the text up to (but not including) the assistant's answer
            # to find where the answer starts in the token sequence
            messages_no_answer = messages[:-1]  # Remove assistant message
            text_no_answer = self.processor.apply_chat_template(
                messages_no_answer, tokenize=False, add_generation_prompt=True
            )
            inputs_no_answer = self.processor(
                text=[text_no_answer],
                images=[image],
                padding=False,
                truncation=True,
                max_length=self.max_seq_length,
                return_tensors="pt",
            )
            prompt_length = inputs_no_answer["input_ids"].shape[1]

            # Mask all prompt tokens with -100
            labels[:prompt_length] = -100
        else:
            # For inference, mask everything
            labels[:] = -100

        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

        if pixel_values is not None:
            result["pixel_values"] = pixel_values
        if image_grid_thw is not None:
            result["image_grid_thw"] = image_grid_thw

        # Store metadata for evaluation
        result["_question"] = sample["question"]
        result["_answer"] = sample["answer"]
        result["_question_type"] = sample.get("question_type", "open")
        result["_dataset"] = sample.get("dataset", "unknown")
        result["_image_path"] = sample["image_path"]

        return result


def collate_fn(batch: List[dict]) -> dict:
    """
    Custom collate function for variable-length sequences.
    Pads input_ids, attention_mask, and labels to the max length in the batch.
    Handles pixel_values and image_grid_thw stacking.
    """
    # Separate metadata from tensor data
    metadata_keys = [k for k in batch[0] if k.startswith("_")]
    metadata = {k: [sample[k] for sample in batch] for k in metadata_keys}

    # Pad sequences
    max_len = max(sample["input_ids"].shape[0] for sample in batch)

    input_ids_padded = []
    attention_mask_padded = []
    labels_padded = []

    pad_token_id = 0  # Qwen uses 0 as padding

    for sample in batch:
        seq_len = sample["input_ids"].shape[0]
        pad_len = max_len - seq_len

        if pad_len > 0:
            input_ids_padded.append(
                torch.cat([sample["input_ids"],
                          torch.full((pad_len,), pad_token_id, dtype=torch.long)])
            )
            attention_mask_padded.append(
                torch.cat([sample["attention_mask"],
                          torch.zeros(pad_len, dtype=torch.long)])
            )
            labels_padded.append(
                torch.cat([sample["labels"],
                          torch.full((pad_len,), -100, dtype=torch.long)])
            )
        else:
            input_ids_padded.append(sample["input_ids"])
            attention_mask_padded.append(sample["attention_mask"])
            labels_padded.append(sample["labels"])

    result = {
        "input_ids": torch.stack(input_ids_padded),
        "attention_mask": torch.stack(attention_mask_padded),
        "labels": torch.stack(labels_padded),
    }

    # Handle pixel_values — these can have variable shapes per image
    if "pixel_values" in batch[0] and batch[0]["pixel_values"] is not None:
        # For Qwen3-VL, pixel_values are typically [num_patches, C, H, W]
        # We concatenate along the first dimension
        pixel_values_list = [s["pixel_values"] for s in batch if s.get("pixel_values") is not None]
        if pixel_values_list:
            result["pixel_values"] = torch.cat(pixel_values_list, dim=0)

    if "image_grid_thw" in batch[0] and batch[0]["image_grid_thw"] is not None:
        grid_thw_list = [s["image_grid_thw"] for s in batch if s.get("image_grid_thw") is not None]
        if grid_thw_list:
            result["image_grid_thw"] = torch.stack(grid_thw_list)

    # Attach metadata
    result["_metadata"] = metadata

    return result
