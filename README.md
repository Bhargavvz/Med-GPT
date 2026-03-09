# 🏥 MedGPT — Medical Visual Question Answering

<p align="center">
  <strong>AI-powered medical image analysis with visual explanations</strong>
</p>

<p align="center">
  <a href="#features">Features</a> •
  <a href="#architecture">Architecture</a> •
  <a href="#installation">Installation</a> •
  <a href="#usage">Usage</a> •
  <a href="#results">Results</a> •
  <a href="#datasets">Datasets</a>
</p>

---

## ✨ Features

- **Vision-Language AI** — Powered by Qwen3-VL-8B fine-tuned on 150K+ medical VQA samples
- **Grad-CAM Heatmaps** — Visual explanations showing where the model focuses
- **Multi-Modality** — X-ray, CT, MRI, Ultrasound, and Pathology slides
- **React + Vite Frontend** — Modern, responsive web interface with glassmorphism design
- **FastAPI Backend** — GPU-accelerated inference with REST API
- **Dashboard** — Interactive training curves, accuracy charts, and metrics visualizations

## 🏗️ Architecture

```
MedGPT/
├── frontend/              # React + Vite frontend
│   ├── src/
│   │   ├── pages/         # Home, Analyze, Dashboard, About
│   │   ├── components/    # Navbar, Footer
│   │   └── styles/        # Global CSS design system
│   └── dist/              # Production build
├── backend/               # FastAPI server
│   └── server.py          # API endpoints + React SPA serving
├── models/                # Model definitions
│   ├── medgpt.py          # MedGPT model class
│   └── explainability.py  # Grad-CAM implementation
├── training/              # Training pipeline
│   ├── train.py           # Pre-training + fine-tuning script
│   ├── evaluate.py        # Evaluation metrics
│   └── visualize.py       # Generate training curves & charts
├── inference/             # Inference utilities
│   └── predict.py         # CLI prediction tool
├── data/                  # Datasets
│   ├── prepare_data.py    # Download & preprocess datasets
│   ├── dataset.py         # PyTorch dataset classes
│   └── processed/         # Processed JSON splits
├── configs/
│   └── config.yaml        # All configuration settings
├── checkpoints/           # Trained model checkpoints
│   └── finetune/
│       └── best_model/    # Best LoRA adapter
└── results/               # Evaluation results & visualizations
    └── figures/           # Training curves, charts, heatmaps
```

## 📊 Results

| Metric | Score |
|--------|-------|
| **Overall Accuracy** | 78.5% |
| **Yes/No Accuracy** | 87.1% |
| **Open-ended Accuracy** | 73.9% |
| **BLEU-1** | 82.8% |
| **ROUGE-L** | 80.6% |
| **Token F1** | 81.2% |

### Training Details

| Parameter | Value |
|-----------|-------|
| Base Model | Qwen3-VL-8B-Instruct |
| Fine-tuning Method | LoRA (rank=64, alpha=128) |
| Trainable Parameters | 210M / 9B (2.3%) |
| Precision | bfloat16 |
| Training Epochs | 3 |
| Hardware | NVIDIA H200 (141GB VRAM) |
| Training Time | ~5.6 hours |

## 📦 Datasets

| Dataset | Samples | Stage |
|---------|---------|-------|
| PMC-VQA | ~140K | Pre-training |
| VQA-RAD | ~3.5K | Fine-tuning |
| SLAKE | ~14K | Fine-tuning |
| PathVQA | ~32K | Fine-tuning |

## 🚀 Installation

### Prerequisites
- Python 3.10+
- NVIDIA GPU with 24GB+ VRAM (48GB+ recommended)
- Node.js 18+ (for frontend)

### Setup

```bash
# Clone from HuggingFace
git lfs install
git clone https://huggingface.co/bhargavvz/MedGPT
cd MedGPT

# Install Python dependencies
pip install -r requirements.txt

# Download datasets
python data/prepare_data.py --datasets all --output_dir data/processed --validate

# Build frontend
cd frontend && npm install && npm run build && cd ..
```

## 💻 Usage

### Web Application
```bash
# Start the server (API + React frontend on port 8000)
python backend/server.py

# Open: http://localhost:8000
```

### CLI Inference
```bash
python inference/predict.py \
    --image path/to/xray.jpg \
    --question "What abnormality is visible?" \
    --adapter_path checkpoints/finetune/best_model
```

### Training
```bash
# Full pipeline (pre-training + fine-tuning)
python training/train.py --stage all

# Fine-tuning only
python training/train.py --stage finetune
```

### Evaluation
```bash
python training/evaluate.py \
    --adapter_path checkpoints/finetune/best_model \
    --test_file data/processed/finetune_test.json \
    --output_file results/eval_results.json
```

### Generate Visualizations
```bash
python training/visualize.py --output_dir results/figures
```

## 🐳 Docker

```bash
docker compose up -d
# Access at http://localhost:8000
```

## 🔗 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/health` | Health check & model status |
| `POST` | `/api/predict` | Image + question → answer + heatmap |
| `POST` | `/api/load` | Load/reload model |
| `GET` | `/api/metrics` | Evaluation results |
| `GET` | `/api/training-history` | Training loss curves |

## 📄 Tech Stack

- **Model**: Qwen3-VL-8B + LoRA
- **Backend**: FastAPI + Uvicorn
- **Frontend**: React + Vite + Framer Motion + Recharts
- **Training**: PyTorch + HuggingFace Transformers
- **Explainability**: Grad-CAM
- **Deployment**: Docker + NVIDIA CUDA

## ⚠️ Disclaimer

MedGPT is a research and educational tool. It is **NOT** intended for clinical diagnosis or medical decision-making. Always consult qualified healthcare professionals for medical advice.

## 📝 License

This project is for educational and research purposes only.

---

<p align="center">
  Built with ❤️ by <strong>Bhargav</strong>
</p>
