#!/usr/bin/env python3
"""
MedGPT Explainability Module
Grad-CAM and Attention Rollout for visual explanations of model predictions.
"""

import os
from typing import Optional, Tuple, List

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping for Qwen3-VL.

    Hooks into the vision encoder's last transformer layer,
    computes gradients of the output w.r.t. feature maps,
    and produces a heatmap showing which image regions influenced the prediction.
    """

    def __init__(self, model, processor, target_layer: str = "last"):
        """
        Args:
            model: The MedGPT model (or its underlying HF model)
            processor: AutoProcessor for the model
            target_layer: Which layer to hook. "last" = last vision encoder layer
        """
        self.model = model
        self.processor = processor
        self.target_layer = target_layer

        self.gradients = None
        self.activations = None
        self.hooks = []

    def _find_vision_layers(self):
        """Find the vision encoder layers in the model."""
        base_model = self.model
        # Unwrap PEFT if needed
        if hasattr(base_model, "base_model"):
            base_model = base_model.base_model
        if hasattr(base_model, "model"):
            base_model = base_model.model

        # Look for vision encoder - Qwen2.5-VL / Qwen3-VL structure
        vision_model = None
        if hasattr(base_model, "visual"):
            vision_model = base_model.visual
        elif hasattr(base_model, "vision_model"):
            vision_model = base_model.vision_model
        elif hasattr(base_model, "model") and hasattr(base_model.model, "visual"):
            vision_model = base_model.model.visual

        if vision_model is None:
            print("[WARN] Could not find vision encoder in model")
            return None

        # Find transformer blocks within the vision model
        blocks = None
        if hasattr(vision_model, "blocks"):
            blocks = vision_model.blocks
        elif hasattr(vision_model, "layers"):
            blocks = vision_model.layers
        elif hasattr(vision_model, "encoder") and hasattr(vision_model.encoder, "layers"):
            blocks = vision_model.encoder.layers

        return blocks

    def _register_hooks(self, target_block):
        """Register forward and backward hooks on the target layer."""
        def forward_hook(module, input, output):
            if isinstance(output, tuple):
                self.activations = output[0].detach()
            else:
                self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            if isinstance(grad_output, tuple):
                self.gradients = grad_output[0].detach()
            else:
                self.gradients = grad_output.detach()

        self.hooks.append(target_block.register_forward_hook(forward_hook))
        self.hooks.append(target_block.register_full_backward_hook(backward_hook))

    def _remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def generate_heatmap(
        self,
        image_path: str,
        question: str,
        knowledge: str = "",
        colormap: str = "jet",
        alpha: float = 0.4,
    ) -> Tuple[Image.Image, Image.Image, np.ndarray]:
        """
        Generate a Grad-CAM heatmap for a given image + question.

        Args:
            image_path: Path to the medical image
            question: The question being asked
            knowledge: Optional medical knowledge context
            colormap: Matplotlib colormap name
            alpha: Overlay transparency

        Returns:
            (overlay_image, heatmap_image, raw_cam) tuple
        """
        # Find vision layers and register hooks
        blocks = self._find_vision_layers()
        if blocks is None:
            # Return original image if we can't find vision layers
            orig = Image.open(image_path).convert("RGB")
            return orig, orig, np.zeros((1, 1))

        target_block = blocks[-1]  # Last layer
        self._register_hooks(target_block)

        try:
            # Build input
            orig_image = Image.open(image_path).convert("RGB")
            w, h = orig_image.size

            user_content = [
                {"type": "image", "image": image_path},
                {"type": "text", "text": f"Question: {question}"},
            ]
            if knowledge:
                user_content[1]["text"] = f"Medical Context: {knowledge}\nQuestion: {question}"

            messages = [
                {"role": "user", "content": user_content},
            ]

            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.processor(
                text=[text],
                images=[orig_image],
                return_tensors="pt",
                padding=True,
            )

            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                      for k, v in inputs.items()}

            # Enable gradients
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad_(False)

            # Forward pass
            inputs["input_ids"].requires_grad_(False)
            outputs = self.model(**inputs, output_hidden_states=True)

            # Get the logits for the most likely next token
            logits = outputs.logits[:, -1, :]
            pred_token = logits.argmax(dim=-1)
            pred_score = logits[0, pred_token[0]]

            # Backward pass
            self.model.zero_grad()
            pred_score.backward(retain_graph=False)

            # Compute Grad-CAM
            if self.gradients is not None and self.activations is not None:
                # Global average pooling of gradients
                weights = self.gradients.mean(dim=1, keepdim=True)  # [B, 1, D]
                cam = (weights * self.activations).sum(dim=-1)  # [B, N]
                cam = F.relu(cam)  # Only positive contributions

                # Normalize
                cam = cam[0]  # First (only) batch element
                cam_min = cam.min()
                cam_max = cam.max()
                if cam_max > cam_min:
                    cam = (cam - cam_min) / (cam_max - cam_min)
                else:
                    cam = torch.zeros_like(cam)

                cam_np = cam.cpu().numpy()

                # Reshape to 2D — assume square grid of patches
                n_patches = cam_np.shape[0]
                grid_size = int(np.sqrt(n_patches))
                if grid_size * grid_size != n_patches:
                    # Non-square: find closest rectangle
                    grid_size = max(1, grid_size)
                    cam_np = cam_np[:grid_size * grid_size]

                cam_2d = cam_np.reshape(grid_size, grid_size)

                # Resize to original image dimensions
                from scipy.ndimage import zoom
                scale_h = h / grid_size
                scale_w = w / grid_size
                cam_resized = zoom(cam_2d, (scale_h, scale_w), order=1)
                cam_resized = np.clip(cam_resized, 0, 1)

                # Create heatmap
                cmap = cm.get_cmap(colormap)
                heatmap_rgba = cmap(cam_resized)
                heatmap_rgb = (heatmap_rgba[:, :, :3] * 255).astype(np.uint8)
                heatmap_image = Image.fromarray(heatmap_rgb)

                # Create overlay
                orig_array = np.array(orig_image).astype(np.float32) / 255.0
                heat_array = heatmap_rgb.astype(np.float32) / 255.0

                # Resize heatmap to match original if needed
                if heat_array.shape[:2] != orig_array.shape[:2]:
                    heatmap_image = heatmap_image.resize((w, h), Image.BILINEAR)
                    heat_array = np.array(heatmap_image).astype(np.float32) / 255.0

                overlay_array = (1 - alpha) * orig_array + alpha * heat_array
                overlay_array = np.clip(overlay_array * 255, 0, 255).astype(np.uint8)
                overlay_image = Image.fromarray(overlay_array)

                return overlay_image, heatmap_image, cam_resized
            else:
                print("[WARN] No gradients captured — returning original image")
                return orig_image, orig_image, np.zeros((1, 1))

        finally:
            self._remove_hooks()

    def save_heatmap(
        self,
        image_path: str,
        question: str,
        output_path: str,
        knowledge: str = "",
        figsize: tuple = (15, 5),
    ):
        """Generate and save Grad-CAM visualization as a figure."""
        overlay, heatmap, cam = self.generate_heatmap(
            image_path, question, knowledge
        )
        orig = Image.open(image_path).convert("RGB")

        fig, axes = plt.subplots(1, 3, figsize=figsize)

        axes[0].imshow(orig)
        axes[0].set_title("Original Image", fontsize=12)
        axes[0].axis("off")

        axes[1].imshow(heatmap)
        axes[1].set_title("Grad-CAM Heatmap", fontsize=12)
        axes[1].axis("off")

        axes[2].imshow(overlay)
        axes[2].set_title("Overlay", fontsize=12)
        axes[2].axis("off")

        plt.suptitle(f"Q: {question[:80]}{'...' if len(question) > 80 else ''}",
                     fontsize=11, y=0.02)
        plt.tight_layout()

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Heatmap saved to {output_path}")


class AttentionRollout:
    """
    Attention Rollout visualization across transformer layers.

    Aggregates attention matrices from all layers to show
    where the model attended in the image.
    """

    def __init__(self, model, processor, head_fusion: str = "mean"):
        """
        Args:
            model: The MedGPT model
            processor: AutoProcessor
            head_fusion: How to combine attention heads ("mean", "max", "min")
        """
        self.model = model
        self.processor = processor
        self.head_fusion = head_fusion

    def generate_attention_map(
        self,
        image_path: str,
        question: str,
        knowledge: str = "",
    ) -> Tuple[Image.Image, np.ndarray]:
        """
        Generate attention rollout visualization.

        Returns:
            (overlay_image, attention_map) tuple
        """
        orig_image = Image.open(image_path).convert("RGB")
        w, h = orig_image.size

        user_content = [
            {"type": "image", "image": image_path},
            {"type": "text", "text": f"Question: {question}"},
        ]
        if knowledge:
            user_content[1]["text"] = f"Medical Context: {knowledge}\nQuestion: {question}"

        messages = [{"role": "user", "content": user_content}]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.processor(
            text=[text],
            images=[orig_image],
            return_tensors="pt",
            padding=True,
        )

        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                  for k, v in inputs.items()}

        # Forward pass with attention output
        with torch.no_grad():
            outputs = self.model(
                **inputs,
                output_attentions=True,
                output_hidden_states=False,
            )

        attentions = outputs.attentions
        if attentions is None or len(attentions) == 0:
            print("[WARN] No attention outputs — model may not support output_attentions")
            return orig_image, np.zeros((1, 1))

        # Aggregate attention across layers
        # Each attention: [B, num_heads, seq_len, seq_len]
        rollout = torch.eye(attentions[0].shape[-1], device=device).unsqueeze(0)

        for attention in attentions:
            # Fuse heads
            if self.head_fusion == "mean":
                attn = attention.mean(dim=1)
            elif self.head_fusion == "max":
                attn = attention.max(dim=1).values
            elif self.head_fusion == "min":
                attn = attention.min(dim=1).values
            else:
                attn = attention.mean(dim=1)

            # Add residual connection and normalize
            attn = attn + torch.eye(attn.shape[-1], device=device).unsqueeze(0)
            attn = attn / attn.sum(dim=-1, keepdim=True)

            # Multiply
            rollout = torch.matmul(attn, rollout)

        # Extract attention to image tokens (from the last token's perspective)
        # The last token attends to all previous tokens including image tokens
        last_token_attn = rollout[0, -1, :]  # [seq_len]

        # We need to identify which tokens are image tokens
        # For now, take the middle portion (image tokens are typically after system prompt
        # and before the question text)
        n_tokens = last_token_attn.shape[0]
        # Rough heuristic: image tokens are a large block in the middle
        # For a more precise approach, we'd need to track token positions
        attn_np = last_token_attn.cpu().numpy()

        # Try to find image token range — look for a contiguous high-attention region
        # Simple approach: take sqrt(n_tokens) patches and reshape
        n_image_tokens = min(n_tokens, 256)  # approximate
        image_attn = attn_np[:n_image_tokens]

        grid_size = int(np.sqrt(n_image_tokens))
        if grid_size > 0:
            image_attn = image_attn[:grid_size * grid_size]
            attn_2d = image_attn.reshape(grid_size, grid_size)

            # Normalize
            attn_min = attn_2d.min()
            attn_max = attn_2d.max()
            if attn_max > attn_min:
                attn_2d = (attn_2d - attn_min) / (attn_max - attn_min)

            # Resize to original image size
            from scipy.ndimage import zoom
            scale_h = h / grid_size
            scale_w = w / grid_size
            attn_resized = zoom(attn_2d, (scale_h, scale_w), order=1)
            attn_resized = np.clip(attn_resized, 0, 1)

            # Create overlay
            cmap = cm.get_cmap("viridis")
            heatmap_rgba = cmap(attn_resized)
            heatmap_rgb = (heatmap_rgba[:, :, :3] * 255).astype(np.uint8)

            orig_array = np.array(orig_image).astype(np.float32) / 255.0
            heat_array = heatmap_rgb.astype(np.float32) / 255.0

            if heat_array.shape[:2] != orig_array.shape[:2]:
                hm = Image.fromarray(heatmap_rgb).resize((w, h), Image.BILINEAR)
                heat_array = np.array(hm).astype(np.float32) / 255.0

            overlay_array = 0.6 * orig_array + 0.4 * heat_array
            overlay_array = np.clip(overlay_array * 255, 0, 255).astype(np.uint8)
            overlay_image = Image.fromarray(overlay_array)

            return overlay_image, attn_resized
        else:
            return orig_image, np.zeros((1, 1))
