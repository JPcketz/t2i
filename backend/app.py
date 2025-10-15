import os, uuid, torch
from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional
from diffusers import StableDiffusionXLPipeline
from PIL import Image

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

@app.post("/generate")
def generate(req: GenRequest, bg: BackgroundTasks):
    job_id = uuid.uuid4().hex
    JOBS[job_id] = {"status": "queued"}
    bg.add_task(run_job, job_id, req)
    return {"job_id": job_id}

@app.get("/jobs/{job_id}")
def job_status(job_id: str):
    return JOBS.get(job_id, {"status": "not_found"})

# Serve generated images
app.mount("/static", StaticFiles(directory=OUT_DIR), name="static")