import imageio
import numpy as np
import cv2
import os, uuid, torch
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler
from diffusers import AutoPipelineForText2Image, LCMScheduler, StableVideoDiffusionPipeline
from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional
from diffusers import StableDiffusionXLPipeline
from PIL import Image

svd_pipe = None

app = FastAPI(title="T2I API")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT_DIR = os.path.abspath(os.getenv("OUT_DIR", "./data"))
os.makedirs(OUT_DIR, exist_ok=True)

# CORS for Next.js dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Tiny in-memory job store (MVP)
JOBS = {}

# Load model once at startup
pipe = None
@app.on_event("startup")
def load_model():
    global pipe
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        use_safetensors=True
    )
    if DEVICE == "cuda":
        pipe = pipe.to(DEVICE)
    # Disable built-in checker for MVP; we’ll add our own basic prompt filter
    pipe.safety_checker = None

class GenRequest(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = ""
    steps: Optional[int] = 30
    cfg: Optional[float] = 6.5
    seed: Optional[int] = None
    width: Optional[int] = 1024
    height: Optional[int] = 1024

class VidRequest(BaseModel):
    prompt: str
    # generation settings for the *keyframe* (we’ll reuse our mode switch)
    mode: Optional[str] = "turbo"   # fast default for video
    steps: Optional[int] = 4
    cfg: Optional[float] = 1.0
    width: Optional[int] = 512
    height: Optional[int] = 512
    seed: Optional[int] = None

    # SVD settings
    num_frames: Optional[int] = 14   # 14 or 24 are common
    fps: Optional[int] = 14
    motion_bucket_id: Optional[int] = 127  # 1..255; higher = more motion
    noise_aug_strength: Optional[float] = 0.02  # small noise helps temporal stability

@app.get("/health")
def health():
    return {"status": "ok", "device": DEVICE}

def basic_blocklist(prompt: str) -> bool:
    blocked = ["illegal", "abuse", "gore"]  # placeholder list; expand later
    p = prompt.lower()
    return any(word in p for word in blocked)

def run_job(job_id: str, req: GenRequest):
    try:
        if basic_blocklist(req.prompt):
            JOBS[job_id]["status"] = "blocked"
            return
        generator = torch.Generator(device=DEVICE)
        if req.seed is not None:
            generator = generator.manual_seed(req.seed)
        result = pipe(
            prompt=req.prompt,
            negative_prompt=req.negative_prompt,
            num_inference_steps=req.steps,
            guidance_scale=req.cfg,
            width=req.width,
            height=req.height,
            generator=generator
        )
        img: Image.Image = result.images[0]
        out_path = os.path.join(OUT_DIR, f"{job_id}.png")
        img.save(out_path)
        JOBS[job_id]["status"] = "done"
        JOBS[job_id]["output"] = {"image_path": f"/static/{job_id}.png"}
    except Exception as e:
        JOBS[job_id]["status"] = "error"
        JOBS[job_id]["error"] = str(e)

def frames_to_mp4(frames, out_path, fps=14):
    # frames: list of PIL Images
    arrs = [cv2.cvtColor(np.array(f), cv2.COLOR_RGB2BGR) for f in frames]
    height, width, _ = arrs[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
    for f in arrs:
        writer.write(f)
    writer.release()

def get_svd_pipe():
    global svd_pipe
    if svd_pipe is None:
        # Image-to-Video: StabilityAI Stable Video Diffusion
        # Good default: 14–24 frames, ~576p wide; runs fast-ish on modern NVIDIA GPUs
        svd_pipe = StableVideoDiffusionPipeline.from_pretrained(
            "stabilityai/stable-video-diffusion-img2vid-xt",
            torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
            use_safetensors=True
        )
        if DEVICE == "cuda":
            svd_pipe = svd_pipe.to(DEVICE)
            try: svd_pipe.enable_xformers_memory_efficient_attention()
            except: pass
            svd_pipe.enable_vae_tiling()
            svd_pipe.unet.to(memory_format=torch.channels_last)
    return svd_pipe

@app.post("/generate")
def generate(req: GenRequest, bg: BackgroundTasks):
    job_id = uuid.uuid4().hex
    JOBS[job_id] = {"status": "queued"}
    bg.add_task(run_job, job_id, req)
    return {"job_id": job_id}

@app.post("/generate_video")
def generate_video(req: VidRequest, bg: BackgroundTasks):
    job_id = uuid.uuid4().hex
    JOBS[job_id] = {"status": "queued"}
    def task():
        try:
            # 1) Create a fast keyframe using our existing mode switch
            generator = torch.Generator(device=DEVICE)
            if req.seed is not None:
                generator = generator.manual_seed(req.seed)

            # choose pipeline
            chosen = pipe
            steps = req.steps or 4
            cfg = req.cfg or 1.0
            width = req.width or 512
            height = req.height or 512

            if req.mode == "fast-lcm":
                chosen = get_fast_pipe()
                steps = min(steps, 8)
                cfg = min(cfg, 2.0)
            elif req.mode == "turbo":
                chosen = get_turbo_pipe()
                steps = min(steps, 4)
                cfg = min(cfg, 2.0)
                width = 512
                height = 512

            key = chosen(
                prompt=req.prompt,
                negative_prompt="",
                num_inference_steps=steps,
                guidance_scale=cfg,
                width=width,
                height=height,
                generator=generator
            ).images[0]

            # 2) Animate with SVD
            svd = get_svd_pipe()
            video = svd(
                key,
                decode_chunk_size=8,  # speed/memory tradeoff
                num_frames=req.num_frames,
                motion_bucket_id=req.motion_bucket_id,
                noise_aug_strength=req.noise_aug_strength
            ).frames[0]  # list of PIL.Image

            # 3) Save MP4
            out_path = os.path.join(OUT_DIR, f"{job_id}.mp4")
            frames_to_mp4(video, out_path, fps=req.fps or 14)

            JOBS[job_id]["status"] = "done"
            JOBS[job_id]["output"] = {"video_path": f"/static/{job_id}.mp4"}
        except Exception as e:
            JOBS[job_id]["status"] = "error"
            JOBS[job_id]["error"] = str(e)

    bg.add_task(task)
    return {"job_id": job_id}

@app.get("/jobs/{job_id}")
def job_status(job_id: str):
    return JOBS.get(job_id, {"status": "not_found"})

# Serve generated images
app.mount("/static", StaticFiles(directory=OUT_DIR), name="static")