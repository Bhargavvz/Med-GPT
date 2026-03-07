# 🏥 MedGPT — Medical Visual Question Answering

A production-grade Medical VQA system that takes a medical image (X-ray, CT, MRI, ultrasound, or pathology slide) and a natural language question, then generates an accurate answer, a medical rationale, and a Grad-CAM heatmap showing which regions the model focused on.

## Architecture

```
Image + Question → Qwen3-VL-8B-Instruct (LoRA) → Answer + Rationale
                                                → Grad-CAM Heatmap
```

The system uses **Qwen3-VL-8B-Instruct** as the base VLM, fine-tuned with LoRA on medical VQA datasets. Medical knowledge is injected via the prompt (RAG-style), not through a separate encoder.

## Features

- **Two-stage training**: PMC-VQA pre-training → VQA-RAD + SLAKE + PathVQA fine-tuning
- **4 datasets** auto-downloaded from HuggingFace (277K+ QA pairs)
- **Grad-CAM** heatmaps for visual explainability
- **Attention Rollout** visualization
- **Web application** with premium dark-theme UI
- **Docker deployment** with GPU support
- **Answer normalization** with synonym mapping for robust evaluation
- **4 metrics**: Accuracy (closed/open), BLEU-1, ROUGE-L, Token F1

## Expected Results

| Metric | Target Range |
|--------|-------------|
| Yes/No Accuracy | 82–90% |
| Open-ended Accuracy | 65–78% |
| Overall Accuracy | 75–85% |
| BLEU-1 | 0.60–0.72 |
| ROUGE-L | 0.63–0.75 |

## Requirements

- **GPU**: NVIDIA H200 (141GB VRAM) recommended. Works with any GPU ≥16GB (with quantization)
- **Python**: 3.10+
- **CUDA**: 12.x

## Quick Start

### 1. Install Dependencies

```bash
cd Med
pip install -r requirements.txt
```

### 2. Prepare Data

```bash
# Download all datasets (VQA-RAD + SLAKE + PathVQA + PMC-VQA)
python data/prepare_data.py --datasets all --output_dir data/processed --validate

# Or start with just VQA-RAD for quick iteration
python data/prepare_data.py --datasets vqa_rad --output_dir data/processed --validate
```

### 3. Train

```bash
# Quick dry-run to verify everything works
python training/train.py --stage finetune --dry_run --batch_size 1

# Full two-stage training
python training/train.py --stage both

# Fine-tune only (skip pre-training)
python training/train.py --stage finetune --num_epochs 5 --batch_size 8

# With custom hyperparameters
python training/train.py \
    --stage finetune \
    --num_epochs 5 \
    --batch_size 8 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-4 \
    --eval_steps 100
```

### 4. Evaluate

```bash
python training/evaluate.py \
    --adapter_path checkpoints/finetune/best_model \
    --test_file data/processed/finetune_test.json \
    --output_file results/eval_results.json
```

### 5. Inference

```bash
# Single image
python inference/predict.py \
    --image path/to/xray.jpg \
    --question "What abnormality is visible?" \
    --adapter_path checkpoints/finetune/best_model

# With Grad-CAM heatmap
python inference/predict.py \
    --image path/to/xray.jpg \
    --question "What abnormality is visible?" \
    --gradcam --gradcam_output heatmap.png

# Batch inference
python inference/predict.py \
    --batch queries.json \
    --output results.json
```

### 6. Web Application

```bash
# Start the web server
MEDGPT_ADAPTER=checkpoints/finetune/best_model python app/server.py

# Open http://localhost:8000 in your browser
```

### 7. Docker Deployment

```bash
docker compose up --build
```

## Project Structure

```
Med/
├── configs/config.yaml          # All hyperparameters
├── data/
│   ├── prepare_data.py          # Download & prepare datasets
│   ├── generate_knowledge.py    # Optional: generate knowledge snippets
│   ├── dataset.py               # PyTorch Dataset with label masking
│   └── processed/               # Processed JSON data files
├── models/
│   ├── medgpt.py                # Qwen3-VL + LoRA model wrapper
│   └── explainability.py        # Grad-CAM & Attention Rollout
├── training/
│   ├── train.py                 # Two-stage training pipeline
│   └── evaluate.py              # Evaluation with all metrics
├── inference/
│   └── predict.py               # CLI inference tool
├── app/
│   ├── server.py                # FastAPI backend
│   └── static/                  # Web UI (HTML/CSS/JS)
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## Datasets

| Dataset | Size | Source | Stage |
|---------|------|--------|-------|
| PMC-VQA | 227K | `xmcmic/PMC-VQA` | Pre-training |
| VQA-RAD | 3.5K | `flaviagiammarino/vqa-rad` | Fine-tuning |
| SLAKE | 14K | `BoKelworker/SLAKE` | Fine-tuning |
| PathVQA | 33K | `flaviagiammarino/path-vqa` | Fine-tuning |

## Key Design Decisions

1. **No separate vision encoder** — Qwen3-VL already has one built in
2. **Prompt-based knowledge injection** instead of separate encoder + fusion
3. **Label masking** — only answer tokens contribute to loss (critical for non-zero accuracy)
4. **bf16 precision** — fp16 causes instability with Qwen models
5. **Answer normalization** — lowercase, strip punctuation, remove articles, map synonyms

## Optional: Knowledge Snippets

Generate medical knowledge for each training sample to boost accuracy by +5-10%:

```bash
# Using OpenAI
OPENAI_API_KEY=sk-... python data/generate_knowledge.py \
    --input_file data/processed/finetune_train.json \
    --output_file data/processed/finetune_train_with_knowledge.json \
    --provider openai
```

## License

Research and educational use only. Not for clinical diagnosis.
