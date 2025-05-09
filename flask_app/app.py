from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
# from transformers import AutoModelForCausalLM, AutoTokenizer
# import torch
# import time

# # Config
# model_name = "gpt2"
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Load model/tokenizer
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
# model.eval()

# # FastAPI setup
# app = FastAPI(title="Medical Chatbot")

# class PromptRequest(BaseModel):
#     symptoms: str
#     question: str

# @app.post("/ask")
# def ask(request: PromptRequest):
#     if not request.symptoms or not request.question:
#         raise HTTPException(status_code=400, detail="Both symptoms and question are required.")

#     prompt = f"Symptoms: {request.symptoms}\nQuestion: {request.question}\nAnswer:"
#     input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

#     start = time.time()
#     output_ids = model.generate(
#         input_ids,
#         max_length=100,
#         temperature=0.7,
#         top_p=0.9,
#         repetition_penalty=1.2
#     )
#     latency = time.time() - start

#     answer = tokenizer.decode(output_ids[0], skip_special_tokens=True)
#     return {
#         "answer": answer.replace(prompt, "").strip(),
#         "latency_sec": latency,
#         "device": device.type
#     }

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from prometheus_client import Histogram, Counter, make_asgi_app
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
import torch, time, os
from typing import Optional  
import time, json, asyncio, aiofiles, httpx
from fastapi import FastAPI, Request, Response
import json, time, asyncio, aiofiles           


# ---------------- Config ----------------
MODEL_NAME  = os.getenv("MODEL_NAME", "gpt2")
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- Model -----------------
tok   = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE).eval()

# ---------------- Prometheus metrics ----
LATENCY = Histogram(
    "chatbot_latency_seconds",
    "Request latency",
    ["endpoint", "slice"]
)
ERRORS = Counter(
    "chatbot_errors_total",
    "Count of errors",
    ["endpoint", "slice"]
)

# ---------------- FastAPI setup ---------
middleware = [
    Middleware(CORSMiddleware, allow_origins=["*"])
]
app = FastAPI(title="Medical Chatbo", middleware=middleware)

# Expose /metrics
app.mount("/metrics", make_asgi_app())

class PromptRequest(BaseModel):
    symptoms: str
    question: str
    slice: Optional[str] = "unspecified"   # optional slice label

def generate_answer(prompt: str) -> str:
    ids = tok(prompt, return_tensors="pt").input_ids.to(DEVICE)
    out = model.generate(ids, max_length=100, temperature=0.7,
                         top_p=0.9, repetition_penalty=1.2)
    return tok.decode(out[0], skip_special_tokens=True)

# -------------- Endpoint ---------------
@app.post("/api/v1/answer")
async def ask(req: PromptRequest, raw_request: Request):
    slice_name = req.slice or "unspecified"
    endpoint   = "/answer"

    if not req.symptoms or not req.question:
        ERRORS.labels(endpoint, slice_name).inc()
        raise HTTPException(400, "symptoms and question required")

    prompt = f"Symptoms: {req.symptoms}\nQuestion: {req.question}\nAnswer:"

    t0 = time.perf_counter()
    try:
        answer = generate_answer(prompt).replace(prompt, "").strip()
    except Exception as e:
        ERRORS.labels(endpoint, slice_name).inc()
        raise HTTPException(500, str(e)) from e
    finally:
        LATENCY.labels(endpoint, slice_name).observe(time.perf_counter() - t0)

    return {
        "answer": answer,
        "device": DEVICE.type
    }

# ... (keep all your imports) ...
import json, time, asyncio, aiofiles           # <- new

LOG_PATH = "../../../online_log.jsonl"                  # one file forever

# unchanged: LATENCY, ERRORS, PromptRequest, etc.

@app.post("/api/v1/answer")
async def ask(req: PromptRequest, raw_request: Request):
    slice_name = req.slice or "unspecified"
    endpoint   = "/answer"

    if not req.symptoms or not req.question:
        ERRORS.labels(endpoint, slice_name).inc()
        raise HTTPException(400, "symptoms and question required")

    prompt = f"Symptoms: {req.symptoms}\nQuestion: {req.question}\nAnswer:"

    t0 = time.perf_counter()
    try:
        answer = generate_answer(prompt).replace(prompt, "").strip()
    except Exception as e:
        ERRORS.labels(endpoint, slice_name).inc()
        raise HTTPException(500, str(e)) from e
    finally:
        LATENCY.labels(endpoint, slice_name).observe(time.perf_counter() - t0)

    # ------------ append JSON to log (non-blocking) --------------------
    record = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "symptoms": req.symptoms,
        "question": req.question,
        "slice": slice_name,
        "answer": answer,
        "latency_ms": round((time.perf_counter() - t0) * 1000, 1)
    }
    asyncio.create_task(write_jsonl(LOG_PATH, record))

    return {"answer": answer, "device": DEVICE.type}


# ---------- helper: async append ---------------------------------------
async def write_jsonl(path: str, obj: dict):
    async with aiofiles.open(path, "a") as f:
        await f.write(json.dumps(obj, ensure_ascii=False) + "\n")

