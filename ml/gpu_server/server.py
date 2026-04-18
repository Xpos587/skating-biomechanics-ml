"""FastAPI inference server for Vast.ai Serverless GPU worker.

Runs on the remote GPU. Receives R2 keys, processes video, returns results.
R2 credentials are passed per-request so the worker does not store cloud credentials.
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

import boto3
from fastapi import FastAPI
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Skating ML GPU Worker")

# Models are at /app/data/models/ inside the container
os.environ.setdefault("PROJECT_ROOT", "/app")


@app.on_event("startup")
async def warmup_gpu():
    """Pre-warm CUDA/cuDNN to eliminate cold-start latency."""
    from skating_ml.device import DeviceConfig

    cfg = DeviceConfig.default()
    if not cfg.is_cuda:
        return
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 2
    # Just importing ort and accessing CUDA provider triggers init
    logging.getLogger(__name__).info("GPU warmup: CUDA initialized")


class ProcessRequest(BaseModel):
    video_r2_key: str
    person_click: dict[str, int] | None = None
    frame_skip: int = 1
    layer: int = 3
    tracking: str = "auto"
    export: bool = True
    ml_flags: dict[str, bool] = {}
    element_type: str | None = None
    # R2 credentials passed per-request (worker doesn't store them)
    r2_endpoint_url: str = ""
    r2_access_key_id: str = ""
    r2_secret_access_key: str = ""
    r2_bucket: str = ""


class ProcessResponse(BaseModel):
    video_r2_key: str
    poses_r2_key: str | None = None
    csv_r2_key: str | None = None
    stats: dict
    metrics: list | None = None
    phases: object | None = None
    recommendations: list | None = None


def _s3(req: ProcessRequest):
    return boto3.client(
        "s3",
        endpoint_url=req.r2_endpoint_url,
        aws_access_key_id=req.r2_access_key_id,
        aws_secret_access_key=req.r2_secret_access_key,
        region_name="auto",
    )


@app.post("/process", response_model=ProcessResponse)
async def process(req: ProcessRequest):
    from src.types import PersonClick
    from src.web_helpers import process_video_pipeline

    s3 = _s3(req)

    with tempfile.TemporaryDirectory() as tmpdir:
        video_local = str(Path(tmpdir) / "input.mp4")
        output_local = str(Path(tmpdir) / "output.mp4")

        logger.info("Downloading video from R2: %s", req.video_r2_key)
        s3.download_file(req.r2_bucket, req.video_r2_key, video_local)

        click = (
            PersonClick(x=req.person_click["x"], y=req.person_click["y"])
            if req.person_click
            else None
        )
        ml = req.ml_flags

        logger.info("Running pipeline (ml_flags=%s)", ml)
        result = process_video_pipeline(
            video_path=video_local,
            person_click=click,
            frame_skip=req.frame_skip,
            layer=req.layer,
            tracking=req.tracking,
            blade_3d=False,
            export=req.export,
            output_path=output_local,
            progress_cb=None,
            cancel_event=None,
            depth=ml.get("depth", False),
            optical_flow=ml.get("optical_flow", False),
            segment=ml.get("segment", False),
            foot_track=ml.get("foot_track", False),
            matting=ml.get("matting", False),
            inpainting=ml.get("inpainting", False),
            element_type=req.element_type,
        )

        out_key = req.video_r2_key.replace("input/", "output/")
        logger.info("Uploading result to R2: %s", out_key)
        s3.upload_file(output_local, req.r2_bucket, out_key)

        poses_key = None
        if result.get("poses_path") and Path(result["poses_path"]).exists():
            poses_key = out_key.replace(".mp4", "_poses.npy")
            s3.upload_file(result["poses_path"], req.r2_bucket, poses_key)

        csv_key = None
        if result.get("csv_path") and Path(result["csv_path"]).exists():
            csv_key = out_key.replace(".mp4", "_biomechanics.csv")
            s3.upload_file(result["csv_path"], req.r2_bucket, csv_key)

        return ProcessResponse(
            video_r2_key=out_key,
            poses_r2_key=poses_key,
            csv_r2_key=csv_key,
            stats=result["stats"],
            metrics=result.get("metrics"),
            phases=result.get("phases"),
            recommendations=result.get("recommendations"),
        )


@app.get("/health")
async def health():
    return {"status": "ok"}
