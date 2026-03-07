#!/usr/bin/env python3
"""
MedGPT Web Application — FastAPI Backend
Upload an image, ask a question, get answer + rationale + Grad-CAM heatmap.
"""

import base64
import io
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
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

app = FastAPI(
    title="MedGPT — Medical Visual Question Answering",
    description="Upload a medical image and ask questions about it.",
    version="1.0.0",
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
        print("The model can be loaded later via /load endpoint")
        traceback.print_exc()


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": medgpt_model is not None,
    }


@app.post("/predict")
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
        raise HTTPException(status_code=503, detail="Model not loaded. Please wait or call /load")

    start_time = time.time()

    # Save uploaded image to temp file
    try:
        contents = await image.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")

        # Save to temp file (model needs a file path)
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
                # Convert overlay to base64
                buf = io.BytesIO()
                overlay.save(buf, format="PNG")
                heatmap_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            except Exception as e:
                print(f"[WARN] Grad-CAM failed: {e}")
                heatmap_b64 = None

        # Convert original image to base64 for response
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
        # Clean up temp file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


@app.post("/load")
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


# Serve static files (web UI)
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")


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

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
