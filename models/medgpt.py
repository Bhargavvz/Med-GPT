#!/usr/bin/env python3
"""
MedGPT Model Wrapper
Qwen3-VL-8B-Instruct with LoRA fine-tuning for Medical VQA.
Uses the VLM's native vision encoder — no separate CLIP/ViT.
"""

import os
from typing import Optional, Dict, Any

import torch
import yaml
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel,
    TaskType,
)
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
)


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


class MedGPT:
    """
    MedGPT: Medical VQA model built on Qwen3-VL-8B-Instruct with LoRA.

    Architecture:
        Image + Prompt → Qwen3-VL (native vision encoder) → LoRA layers → Answer

    Key decisions:
        - No separate vision encoder (VLM has its own)
        - No separate knowledge encoder (knowledge injected via prompt)
        - bf16 precision (fp16 causes instability with Qwen)
        - LoRA on all linear layers for maximum expressiveness
    """

    def __init__(
        self,
        config: Optional[dict] = None,
        config_path: str = "configs/config.yaml",
        training: bool = True,
    ):
        if config is None:
            config = load_config(config_path)

        self.config = config
        self.model_name = config["model"]["name"]
        self.training = training

        # Load processor (tokenizer + image processor)
        self.processor = AutoProcessor.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )

        # Load base model
        self.model = self._load_base_model()

        if training:
            # Apply LoRA
            self.model = self._apply_lora()

        print(f"MedGPT initialized: {self.model_name}")
        print(f"  Training mode: {training}")
        print(f"  Trainable params: {self._count_params()}")

    def _load_base_model(self):
        """Load the base Qwen3-VL model."""
        model_config = self.config["model"]

        # Determine dtype
        dtype_str = model_config.get("torch_dtype", "bfloat16")
        torch_dtype = getattr(torch, dtype_str, torch.bfloat16)

        # Quantization config (if specified)
        quant_config = None
        quant_setting = model_config.get("quantization")
        if quant_setting == "4bit":
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_use_double_quant=True,
            )
        elif quant_setting == "8bit":
            quant_config = BitsAndBytesConfig(load_in_8bit=True)

        # Load model
        load_kwargs = {
            "pretrained_model_name_or_path": self.model_name,
            "torch_dtype": torch_dtype,
            "device_map": model_config.get("device_map", "auto"),
            "trust_remote_code": model_config.get("trust_remote_code", True),
        }

        if quant_config:
            load_kwargs["quantization_config"] = quant_config

        # Try Qwen2.5-VL first (Qwen3-VL may use same class or newer)
        try:
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(**load_kwargs)
        except Exception:
            from transformers import AutoModelForVision2Seq
            model = AutoModelForVision2Seq.from_pretrained(**load_kwargs)

        # Prepare for k-bit training if quantized
        if quant_config and self.training:
            model = prepare_model_for_kbit_training(model)

        return model

    def _apply_lora(self):
        """Apply LoRA adapters to the model."""
        lora_config = self.config["lora"]

        peft_config = LoraConfig(
            r=lora_config.get("rank", 64),
            lora_alpha=lora_config.get("alpha", 128),
            lora_dropout=lora_config.get("dropout", 0.05),
            target_modules=lora_config.get("target_modules", "all-linear"),
            task_type=TaskType.CAUSAL_LM,
            bias=lora_config.get("bias", "none"),
        )

        model = get_peft_model(self.model, peft_config)
        model.print_trainable_parameters()

        return model

    def _count_params(self) -> str:
        """Count trainable parameters."""
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        pct = 100 * trainable / total if total > 0 else 0
        return f"{trainable:,} / {total:,} ({pct:.2f}%)"

    def save_adapter(self, output_dir: str):
        """Save LoRA adapter weights."""
        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir)
        self.processor.save_pretrained(output_dir)
        print(f"Adapter saved to {output_dir}")

    @classmethod
    def from_adapter(
        cls,
        adapter_path: str,
        config: Optional[dict] = None,
        config_path: str = "configs/config.yaml",
    ) -> "MedGPT":
        """Load a trained model from LoRA adapter checkpoint."""
        instance = cls.__new__(cls)

        if config is None:
            config = load_config(config_path)

        instance.config = config
        instance.model_name = config["model"]["name"]
        instance.training = False

        # Load processor from adapter dir if available, else from base model
        processor_path = adapter_path if os.path.exists(
            os.path.join(adapter_path, "preprocessor_config.json")
        ) else instance.model_name

        instance.processor = AutoProcessor.from_pretrained(
            processor_path,
            trust_remote_code=True,
        )

        # Load base model
        instance.model = instance._load_base_model()

        # Load LoRA adapter
        instance.model = PeftModel.from_pretrained(
            instance.model,
            adapter_path,
            is_trainable=False,
        )
        instance.model.eval()

        print(f"MedGPT loaded from adapter: {adapter_path}")
        return instance

    def generate(
        self,
        image_path: str,
        question: str,
        knowledge: str = "",
        max_new_tokens: int = 256,
        temperature: float = 0.1,
        do_sample: bool = False,
        **kwargs,
    ) -> str:
        """
        Generate an answer for a medical image + question.

        Args:
            image_path: Path to the medical image
            question: Natural language question about the image
            knowledge: Optional medical knowledge context
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            do_sample: Whether to use sampling

        Returns:
            Generated answer text
        """
        from PIL import Image as PILImage

        # Build messages
        user_content = [
            {"type": "image", "image": image_path},
        ]

        text_parts = []
        if knowledge:
            text_parts.append(f"Medical Context: {knowledge}")
        text_parts.append(f"Question: {question}")

        user_content.append({"type": "text", "text": "\n".join(text_parts)})

        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert medical imaging specialist with extensive training in "
                    "radiology, pathology, and clinical diagnosis. Analyze the provided medical "
                    "image carefully and answer the question accurately and concisely. "
                    "Provide a brief medical rationale for your answer when appropriate."
                ),
            },
            {"role": "user", "content": user_content},
        ]

        # Process
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image = PILImage.open(image_path).convert("RGB")

        inputs = self.processor(
            text=[text],
            images=[image],
            return_tensors="pt",
            padding=True,
        )

        # Move to model device
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                  for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if do_sample else None,
                do_sample=do_sample,
                **kwargs,
            )

        # Decode only the generated tokens (after the input)
        input_len = inputs["input_ids"].shape[1]
        generated = output_ids[0, input_len:]
        answer = self.processor.decode(generated, skip_special_tokens=True)

        return answer.strip()

    def get_vision_features(self, image_path: str) -> dict:
        """
        Extract intermediate vision encoder features for Grad-CAM.
        Returns a dict with feature maps and the model for gradient computation.
        """
        from PIL import Image as PILImage

        image = PILImage.open(image_path).convert("RGB")

        # Create a minimal input to get vision features
        messages = [
            {"role": "user", "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": "Describe this image."},
            ]},
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.processor(
            text=[text],
            images=[image],
            return_tensors="pt",
            padding=True,
        )

        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                  for k, v in inputs.items()}

        return inputs, image
