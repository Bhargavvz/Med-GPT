#!/usr/bin/env python3
"""
MedGPT Web Application — FastAPI Backend
Serves the React frontend + API endpoints for prediction, metrics, and health checks.
"""

import base64
import io
import json
import os
import sys
import tempfile
import time
import traceback
from pathlib import Path

import uvicorn
import yaml
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

app = FastAPI(
    title="MedGPT — Medical Visual Question Answering",
    description="Upload a medical image and ask questions about it.",
    version="2.0.0",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model reference
medgpt_model = None
gradcam_engine = None


def load_model():
    """Load the MedGPT model at startup."""
    global medgpt_model, gradcam_engine

    from models.medgpt import MedGPT, load_config
    from models.explainability import GradCAM

    config_path = os.environ.get("MEDGPT_CONFIG", "configs/config.yaml")
    adapter_path = os.environ.get("MEDGPT_ADAPTER", None)

    config = load_config(config_path)
    if adapter_path is None:
        adapter_path = config["inference"]["adapter_path"]

    print(f"Loading MedGPT model from: {adapter_path}")
    medgpt_model = MedGPT.from_adapter(adapter_path, config=config)

    # Initialize Grad-CAM
    gradcam_engine = GradCAM(medgpt_model.model, medgpt_model.processor)
    print("MedGPT model loaded and ready!")


@app.on_event("startup")
async def startup_event():
    """Load model when the server starts."""
    try:
        load_model()
    except Exception as e:
        print(f"[WARN] Failed to load model at startup: {e}")
        print("The model can be loaded later via /api/load endpoint")
        traceback.print_exc()


# =========================================================================
# API Routes (prefixed with /api/)
# =========================================================================

@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": medgpt_model is not None,
    }


@app.post("/api/predict")
async def predict(
    image: UploadFile = File(...),
    question: str = Form(...),
    knowledge: str = Form(default=""),
    generate_heatmap: bool = Form(default=True),
    max_new_tokens: int = Form(default=256),
):
    """
    Main prediction endpoint.
    Accepts an image and question, returns answer + explanation + heatmap.
    """
    if medgpt_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Please wait or call /api/load")

    start_time = time.time()

    # Save uploaded image to temp file
    try:
        contents = await image.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            img.save(tmp, "JPEG", quality=95)
            tmp_path = tmp.name
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    try:
        # Generate answer
        answer = medgpt_model.generate(
            image_path=tmp_path,
            question=question,
            knowledge=knowledge,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

        # Generate heatmap
        heatmap_b64 = None
        if generate_heatmap and gradcam_engine:
            try:
                overlay, heatmap_img, cam = gradcam_engine.generate_heatmap(
                    image_path=tmp_path,
                    question=question,
                    knowledge=knowledge,
                )
                buf = io.BytesIO()
                overlay.save(buf, format="PNG")
                heatmap_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            except Exception as e:
                print(f"[WARN] Grad-CAM failed: {e}")
                heatmap_b64 = None

        # Convert original image to base64
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        original_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        elapsed = time.time() - start_time

        return JSONResponse({
            "answer": answer,
            "question": question,
            "knowledge": knowledge,
            "heatmap": heatmap_b64,
            "original_image": original_b64,
            "processing_time": round(elapsed, 2),
        })

    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


@app.post("/api/load")
async def load_model_endpoint(
    adapter_path: str = Form(default=None),
    config_path: str = Form(default="configs/config.yaml"),
):
    """Load or reload the model."""
    global medgpt_model, gradcam_engine

    try:
        if adapter_path:
            os.environ["MEDGPT_ADAPTER"] = adapter_path
        if config_path:
            os.environ["MEDGPT_CONFIG"] = config_path

        load_model()
        return {"status": "Model loaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {e}")


@app.get("/api/metrics")
async def get_metrics():
    """Serve evaluation results."""
    project_root = Path(__file__).parent.parent
    results_path = project_root / "results" / "eval_results.json"

    if not results_path.exists():
        # Also check checkpoints directory
        alt_path = project_root / "checkpoints" / "finetune" / "eval_results.json"
        if alt_path.exists():
            results_path = alt_path
        else:
            raise HTTPException(status_code=404, detail="No evaluation results found. Run evaluate.py first.")

    with open(results_path) as f:
        data = json.load(f)

    # Return just metrics (not all predictions)
    return data.get("metrics", data)


@app.get("/api/training-history")
async def get_training_history():
    """Serve trainer_state.json for training curves."""
    project_root = Path(__file__).parent.parent

    # Find trainer_state.json in checkpoint dirs
    for ckpt in ["checkpoint-2472", "checkpoint-2000", "checkpoint-1500"]:
        path = project_root / "checkpoints" / "finetune" / ckpt / "trainer_state.json"
        if path.exists():
            with open(path) as f:
                return json.load(f)

    raise HTTPException(status_code=404, detail="No training history found.")


# =========================================================================
# Serve React Frontend (must be LAST)
# =========================================================================

# Serve React build
frontend_dist = Path(__file__).parent.parent / "frontend" / "dist"
if frontend_dist.exists():
    # Serve static assets
    app.mount("/assets", StaticFiles(directory=str(frontend_dist / "assets")), name="assets")

    # Catch-all for React Router (SPA)
    @app.get("/{path:path}")
    async def serve_react(path: str):
        """Serve React app — fallback to index.html for client-side routing."""
        file_path = frontend_dist / path
        if file_path.is_file():
            return FileResponse(str(file_path))
        return FileResponse(str(frontend_dist / "index.html"))
else:
    print("[INFO] No React build found at frontend/dist/. Run: cd frontend && npm run build")


def main():
    """Run the MedGPT web server."""
    config_path = os.environ.get("MEDGPT_CONFIG", "configs/config.yaml")
    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)
        host = config.get("app", {}).get("host", "0.0.0.0")
        port = config.get("app", {}).get("port", 8000)
    except Exception:
        host = "0.0.0.0"
        port = 8000

    print(f"\n🚀 MedGPT server starting on http://{host}:{port}")
    if frontend_dist.exists():
        print(f"📦 Serving React build from: {frontend_dist}")
    print()

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
