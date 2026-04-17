# ============================================================
# Medical Document Understanding API (Production + Streaming)
# FastAPI + Gradio + Streaming + Robust Inference + Enhanced Preprocessing
# Updated: Auto-Orientation + Deskew + Contrast Enhancement + 1024px Min Image
# ============================================================
#!/usr/bin/env python3
import os, torch, json, re, gc, logging, io, numpy as np
from threading import Thread
from PIL import Image, ImageOps, ImageEnhance
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from transformers import (
    AutoProcessor, AutoModelForImageTextToText,
    BitsAndBytesConfig, TextIteratorStreamer
)
import gradio as gr

# CONFIG
MODEL_ID = "Sebukpor/medical-document-understanding-v2"
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    logging.warning("⚠️ HF_TOKEN not found. Private repo access may fail.")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cuda.matmul.allow_tf32 = True
MAX_CONTEXT_TOKENS = 10048
MAX_NEW_TOKENS = 6048
MIN_IMAGE_SIZE = 1024

PREPROCESS_CONFIG = {
    "auto_rotate": True,
    "deskew": True,
    "deskew_threshold": 1.5,
    "contrast_factor": 1.25,
    "brightness_threshold": 128,
    "brightness_factor": 1.1,
    "sharpness_factor": 1.0,
    "denoise": False,
}

# PRECISION AUTO-SELECTION
def get_precision_config():
    if DEVICE == "cuda":
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            logger.info("🚀 Using BF16 precision (A100/L40/H100)")
            return {"torch_dtype": torch.bfloat16, "quantization_config": None}
        else:
            logger.info("⚡ Using INT8 quantization (T4/V100 fallback)")
            return {
                "torch_dtype": torch.float16,
                "quantization_config": BitsAndBytesConfig(load_in_8bit=True)
            }
    logger.warning("🐌 Running on CPU")
    return {"torch_dtype": torch.float32, "quantization_config": None}

precision_config = get_precision_config()

# LOAD MODEL & PROCESSOR
def load_components():
    load_kwargs = {"token": HF_TOKEN, "trust_remote_code": True}
    logger.info("📥 Loading processor...")
    processor = AutoProcessor.from_pretrained(MODEL_ID, **load_kwargs)
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id
    logger.info("📥 Loading model...")
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, device_map="auto", low_cpu_mem_usage=True,
        token=HF_TOKEN, trust_remote_code=True, **precision_config
    )
    model.eval()
    logger.info("✅ Model & processor loaded")
    return processor, model

processor, model = load_components()

# SYSTEM PROMPT
SYSTEM_PROMPT = (
    "### ROLE\nYou are an expert Medical Transcriptionist. "
    "Digitize handwritten/printed OPD reports into strict JSON.\n\n"
    "### TASK\nExtract all printed & handwritten info into the schema below.\n\n"
    "### RULES\n"
    "1. Combine clinical notes into single strings.\n"
    "2. Preserve medical shorthand exactly.\n"
    "3. Split prescriptions into separate objects.\n"
    "4. Booleans: consent_given=true only if Yes/signed, signature_present=true if visible.\n"
    '5. Use "" or null for missing data. Use "[unclear]" for illegible text.\n'
    "6. Reassign info to correct fields if misplaced.\n\n"
    "### SCHEMA\nReturn ONLY valid JSON (no markdown fences):\n"
    '{"patient_demographics":{"patient_name":"","father_husband_name":"","age_sex":"","phone_number":"","address":"","appointment_date":"","uhid":"","bill_no":"","dept":""},'
    '"clinical_notes":{"presenting_complaints":"","history_of_allergies":"","examination_findings":"","investigations_advised":"","result_of_investigations":"","provisional_diagnosis":"","nutritional_needs":""},'
    '"procedures":{"consent_given":false,"procedure_note":""},'
    '"prescription":[{"drug_dose":"","route":"","frequency":"","duration":""}],'
    '"follow_up":{"date":"","urgent_care_instructions":""},'
    '"doctor_details":{"signature_present":false,"name_stamp":""}}'
)

# IMAGE PREPROCESSING
def preprocess_image(image: Image.Image) -> Image.Image:
    try:
        image = ImageOps.exif_transpose(image)
        if image.mode != "RGB":
            image = image.convert("RGB")

        if PREPROCESS_CONFIG.get("auto_rotate", True):
            w, h = image.size
            if h / w < 0.7:
                image = image.rotate(90, expand=True, resample=Image.Resampling.BICUBIC)

        if PREPROCESS_CONFIG.get("deskew", True):
            try:
                import cv2
                from deskew import determine_skew
                img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
                angle = determine_skew(gray)
                if abs(angle) > PREPROCESS_CONFIG.get("deskew_threshold", 1.5):
                    image = image.rotate(angle, resample=Image.Resampling.BICUBIC, expand=True, fillcolor=(255, 255, 255))
            except Exception as e:
                logger.debug(f"Deskew skipped: {e}")

        if min(image.size) < MIN_IMAGE_SIZE:
            scale = MIN_IMAGE_SIZE / min(image.size)
            image = image.resize((int(image.width * scale), int(image.height * scale)), Image.Resampling.LANCZOS)

        cf = PREPROCESS_CONFIG.get("contrast_factor", 1.25)
        if cf != 1.0:
            image = ImageEnhance.Contrast(image).enhance(cf)

        bf = PREPROCESS_CONFIG.get("brightness_factor", 1.1)
        if np.mean(np.array(image)) < PREPROCESS_CONFIG.get("brightness_threshold", 128) and bf != 1.0:
            image = ImageEnhance.Brightness(image).enhance(bf)

        return image
    except Exception as e:
        logger.warning(f"Preprocess fallback: {e}")
        return image.convert("RGB") if image.mode != "RGB" else image

# BUILD MESSAGES
def build_messages(pil_image: Image.Image) -> list:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "image", "image": pil_image},
            {"type": "text", "text": "Extract all medical information into structured JSON."}
        ]}
    ]

# ✅ FIXED: Robustly extracts the COMPLETE JSON object using brace-balancing
def parse_json_robust(text: str) -> dict:
    if not text:
        return {"error": "Empty response", "raw_output": ""}
    
    # Remove markdown code fences
    text = re.sub(r"```json\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"```\s*$", "", text, flags=re.IGNORECASE).strip()
    
    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # Find the FIRST '{' and balance braces to extract the full JSON
    start_idx = text.find('{')
    if start_idx == -1:
        logger.warning(f"No JSON object found. Preview: {text[:200]}...")
        return {"error": "No JSON object found", "raw_output": text}
    
    depth = 0
    end_idx = None
    for i, char in enumerate(text[start_idx:], start_idx):
        if char == '{':
            depth += 1
        elif char == '}':
            depth -= 1
            if depth == 0:
                end_idx = i + 1
                break
                
    if end_idx is None:
        logger.warning(f"Unbalanced JSON braces. Preview: {text[:300]}...")
        return {"error": "Unbalanced JSON", "raw_output": text}
    
    json_str = text[start_idx:end_idx]
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        logger.warning(f"JSON parse failed: {e}")
        return {"error": "JSON parsing failed", "raw_output": json_str}

# ✅ FIXED: Handles splitting even when model outputs reasoning WITHOUT tags
def split_thinking_and_json(raw_text: str):
    thinking = ""
    json_part = ""
    
    # 1. Check for explicit thinking tags first
    close_tag_match = re.search(r"</think>", raw_text, re.IGNORECASE)
    if close_tag_match:
        thinking = raw_text[:close_tag_match.start()].strip()
        json_part = raw_text[close_tag_match.end():].strip()
        thinking = re.sub(r"^\s*<think>\s*", "", thinking, flags=re.IGNORECASE).strip()
    else:
        # 2. If no tags, check for opening tag
        open_tag_match = re.search(r"<think>", raw_text, re.IGNORECASE)
        if open_tag_match:
            thinking = raw_text[open_tag_match.end():].strip()
            json_part = ""
        else:
            # 3. No tags at all: Heuristic to find the LAST balanced JSON block
            thinking = raw_text.strip()
            json_part = ""
            
            # Find the LAST '{' to see if JSON is at the end
            last_brace = thinking.rfind('{')
            if last_brace != -1:
                candidate = thinking[last_brace:]
                depth = 0
                end_idx = -1
                for i, char in enumerate(candidate):
                    if char == '{': depth += 1
                    elif char == '}':
                        depth -= 1
                        if depth == 0:
                            end_idx = i + 1
                            break
                
                if end_idx != -1:
                    try:
                        json.loads(candidate[:end_idx])
                        json_part = candidate[:end_idx]
                        thinking = thinking[:last_brace].strip()
                    except json.JSONDecodeError:
                        pass # Keep as thinking until valid JSON forms

    # Clean markdown fences from json_part
    json_part = re.sub(r"```json\s*", "", json_part, flags=re.IGNORECASE)
    json_part = re.sub(r"```\s*", "", json_part, flags=re.IGNORECASE).strip()
    
    # Pretty print if valid (safe for both full and partial if valid)
    try:
        parsed = json.loads(json_part)
        json_part = json.dumps(parsed, indent=2)
    except Exception:
        pass  # Return raw string if partial/invalid
        
    return thinking, json_part

# STANDARD INFERENCE
@torch.no_grad()
def run_inference(image: Image.Image) -> dict:
    try:
        image = preprocess_image(image)
        msgs = build_messages(image)
        prompt = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[prompt], images=[image], return_tensors="pt", padding=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        outputs = model.generate(
            **inputs, max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False, temperature=0.1, repetition_penalty=1.15
        )
        return parse_json_robust(processor.batch_decode(outputs, skip_special_tokens=True)[0])
    except Exception as e:
        logger.error(f"Inference error: {e}", exc_info=True)
        return {"error": str(e), "raw_output": None}

# STREAMING INFERENCE (yields structured NDJSON chunks)
@torch.no_grad()
def run_inference_stream(image: Image.Image):
    """
    Generator that yields newline-delimited JSON (NDJSON) chunks.
    Each chunk contains: {thinking, json_output, status, progress, error}
    """
    try:
        image = preprocess_image(image)
        msgs = build_messages(image)
        prompt = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[prompt], images=[image], return_tensors="pt", padding=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        streamer = TextIteratorStreamer(
            processor.tokenizer, skip_special_tokens=True,
            skip_prompt=True, timeout=30.0
        )
        
        thread = Thread(target=model.generate, kwargs={
            **inputs, "max_new_tokens": MAX_NEW_TOKENS, "streamer": streamer,
            "do_sample": False, "temperature": 0.1, "repetition_penalty": 1.15,
            "pad_token_id": processor.tokenizer.pad_token_id
        })
        thread.start()
        
        full_text = ""
        last_thinking_len = 0
        last_json_len = 0
        tokens_received = 0
        
        for token in streamer:
            full_text += token
            tokens_received += 1
            
            thinking, json_part = split_thinking_and_json(full_text)
            
            # Only yield if there's NEW content to avoid repetition/flood
            new_thinking = len(thinking) > last_thinking_len
            new_json = len(json_part) > last_json_len
            
            if not (new_thinking or new_json):
                continue
                
            # Determine status
            if json_part and json_part.rstrip().endswith("}"):
                try:
                    json.loads(json_part)
                    status = "complete"
                except json.JSONDecodeError:
                    status = "generating"
            else:
                status = "thinking" if not json_part else "generating"
            
            progress = min(1.0, round(tokens_received / MAX_NEW_TOKENS, 2))
            
            chunk = {
                "thinking": thinking if new_thinking else None,
                "json_output": json_part if new_json else None,
                "status": status,
                "progress": progress,
                "error": None
            }
            yield json.dumps(chunk, ensure_ascii=False) + "\n"
            
            if thinking: last_thinking_len = len(thinking)
            if json_part: last_json_len = len(json_part)
        
        thread.join()
        
        # Final guaranteed complete chunk
        final_thinking, final_json = split_thinking_and_json(full_text)
        yield json.dumps({
            "thinking": final_thinking or None,
            "json_output": final_json or None,
            "status": "complete",
            "progress": 1.0,
            "error": None
        }, ensure_ascii=False) + "\n"
        
    except Exception as e:
        logger.error(f"Stream error: {e}", exc_info=True)
        yield json.dumps({
            "thinking": None, "json_output": None, "status": "error", "progress": 0.0, "error": str(e)
        }, ensure_ascii=False) + "\n"

# FASTAPI APP
app = FastAPI(
    title="Medical Document Understanding API",
    version="1.3.0",
    docs_url="/docs"
)

# CORS for browser integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 🔒 Restrict to specific domains in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", tags=["Health"])
async def health_check():
    return {"status": "healthy", "model": MODEL_ID, "device": DEVICE}

@app.post("/predict", tags=["Inference"])
async def predict(file: UploadFile = File(...)):
    """Non-streaming endpoint: returns complete parsed JSON result"""
    try:
        if not file.content_type or not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Invalid file type. Expected image/*")
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Empty file")
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        result = run_inference(image)
        return JSONResponse(content=result)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Predict endpoint error: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"error": str(e), "raw_output": None})
    finally:
        if hasattr(file, 'file') and file.file:
            file.file.close()

@app.post("/predict_stream", tags=["Inference"])
async def predict_stream(file: UploadFile = File(...)):
    """
    Streaming endpoint returning NDJSON (newline-delimited JSON).
    Each line contains: {thinking, json_output, status, progress, error}
    """
    try:
        if not file.content_type or not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Invalid file type. Expected image/*")
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Empty file")
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        return StreamingResponse(
            run_inference_stream(image),
            media_type="application/x-ndjson",
            headers={
                "X-Accel-Buffering": "no",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Transfer-Encoding": "chunked"
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Stream endpoint error: {e}", exc_info=True)
        error_chunk = json.dumps({
            "thinking": None, "json_output": None, "status": "error", "progress": 0.0, "error": str(e)
        }, ensure_ascii=False) + "\n"
        return StreamingResponse(iter([error_chunk]), media_type="application/x-ndjson", status_code=500)
    finally:
        if hasattr(file, 'file') and file.file:
            file.file.close()


# ─────────────────────────────────────────────────────────────
#  GRADIO UI (UNCHANGED - WORKING PERFECTLY)
# ─────────────────────────────────────────────────────────────

CUSTOM_CSS = """
/* ── Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=DM+Sans:wght@300;400;600&display=swap');

/* ── Root palette ── */
:root {
    --bg:        #0d1117;
    --surface:   #161b22;
    --surface2:  #21262d;
    --border:    #30363d;
    --accent:    #3fb950;
    --accent2:   #58a6ff;
    --warn:      #d29922;
    --text:      #e6edf3;
    --text-dim:  #8b949e;
    --radius:    10px;
    --mono:      'IBM Plex Mono', monospace;
    --sans:      'DM Sans', sans-serif;
}

/* ── Global reset ── */
body, .gradio-container {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: var(--sans) !important;
}

/* ── Header banner ── */
#header-banner {
    background: linear-gradient(135deg, #0d1117 0%, #161b22 60%, #1a2332 100%);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 24px 32px;
    margin-bottom: 20px;
    position: relative;
    overflow: hidden;
}
#header-banner::before {
    content: '';
    position: absolute;
    top: -40%;  left: -10%;
    width: 420px; height: 420px;
    background: radial-gradient(circle, rgba(63,185,80,.12) 0%, transparent 70%);
    pointer-events: none;
}
#header-title {
    font-size: 1.75rem;
    font-weight: 600;
    color: var(--accent);
    font-family: var(--mono);
    letter-spacing: -0.5px;
    margin: 0 0 6px;
}
#header-sub {
    color: var(--text-dim);
    font-size: 0.85rem;
    font-family: var(--mono);
    margin: 0;
}

/* ── Panels ── */
.panel-box {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 18px;
}
.panel-label {
    font-family: var(--mono);
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: var(--text-dim);
    margin-bottom: 10px;
    display: flex;
    align-items: center;
    gap: 8px;
}
.panel-label span.dot {
    width: 7px; height: 7px;
    border-radius: 50%;
    display: inline-block;
}
.dot-green  { background: var(--accent); }
.dot-blue   { background: var(--accent2); }
.dot-yellow { background: var(--warn); }

/* ── Image preview wrapper ── */
#img-preview-wrap {
    position: relative;
    overflow: hidden;
    border-radius: 8px;
    background: var(--surface2);
    min-height: 340px;
    display: flex;
    align-items: center;
    justify-content: center;
}
#img-preview-wrap img {
    max-width: 100%;
    max-height: 520px;
    border-radius: 6px;
    transition: transform 0.25s cubic-bezier(.4,0,.2,1);
    transform-origin: center center;
}

/* ── Toolbar buttons ── */
.toolbar {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    align-items: center;
    margin-bottom: 12px;
}
.toolbar button {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: 7px !important;
    font-family: var(--mono) !important;
    font-size: 0.78rem !important;
    padding: 6px 14px !important;
    cursor: pointer !important;
    transition: background .15s, border-color .15s, transform .1s !important;
    display: flex !important;
    align-items: center !important;
    gap: 5px !important;
}
.toolbar button:hover {
    background: var(--surface) !important;
    border-color: var(--accent) !important;
    color: var(--accent) !important;
    transform: translateY(-1px) !important;
}
.toolbar button:active { transform: translateY(0) !important; }
.toolbar .sep {
    width: 1px; height: 24px;
    background: var(--border);
    margin: 0 4px;
}

/* ── Submit button ── */
#submit-btn {
    background: linear-gradient(135deg, #238636, #2ea043) !important;
    border: none !important;
    color: #fff !important;
    font-family: var(--mono) !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    border-radius: 8px !important;
    padding: 10px 28px !important;
    cursor: pointer !important;
    letter-spacing: 0.5px !important;
    transition: opacity .2s, transform .1s !important;
    width: 100% !important;
    margin-top: 10px !important;
}
#submit-btn:hover { opacity: .88 !important; transform: translateY(-1px) !important; }
#submit-btn:active { transform: translateY(0) !important; }

/* ── Clear button ── */
#clear-btn {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-dim) !important;
    font-family: var(--mono) !important;
    font-size: 0.82rem !important;
    border-radius: 8px !important;
    padding: 8px 20px !important;
    width: 100% !important;
    margin-top: 6px !important;
    cursor: pointer !important;
    transition: border-color .15s, color .15s !important;
}
#clear-btn:hover { border-color: #f85149 !important; color: #f85149 !important; }

/* ── Textboxes ── */
textarea, .gr-textbox textarea {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    font-family: var(--mono) !important;
    font-size: 0.82rem !important;
    border-radius: 8px !important;
    line-height: 1.6 !important;
    resize: vertical !important;
}
textarea:focus { border-color: var(--accent2) !important; outline: none !important; }

/* ── Thinking box special tint ── */
#thinking-box textarea {
    border-color: #553c0a !important;
    background: #1a1500 !important;
}
/* ── JSON box special tint ── */
#json-box textarea {
    border-color: #1b4332 !important;
    background: #0b1f15 !important;
}

/* ── Status bar ── */
#status-bar {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 8px 14px;
    font-family: var(--mono);
    font-size: 0.75rem;
    color: var(--text-dim);
    margin-top: 10px;
    display: flex;
    gap: 18px;
    flex-wrap: wrap;
}
#status-bar .sb-item { display: flex; gap: 5px; align-items: center; }
#status-bar .sb-val { color: var(--accent); font-weight: 600; }

/* ── Gradio overrides ── */
.gradio-container .label-wrap { display: none !important; }
footer { display: none !important; }
.gr-form { background: transparent !important; border: none !important; }
"""

# ── State helpers ── #

def rotate_image(image, angle_state):
    if image is None:
        return None, angle_state
    new_angle = (angle_state + 90) % 360
    rotated = image.rotate(-90, expand=True, resample=Image.Resampling.BICUBIC)
    return rotated, new_angle

def rotate_image_ccw(image, angle_state):
    if image is None:
        return None, angle_state
    new_angle = (angle_state - 90) % 360
    rotated = image.rotate(90, expand=True, resample=Image.Resampling.BICUBIC)
    return rotated, new_angle

def reset_rotation(image_orig, angle_state):
    """Reset to original uploaded image (stored before any rotation)."""
    return image_orig, 0

def gradio_predict_split(image):
    """Stream model output and split into thinking / JSON parts in real time."""
    if image is None:
        yield "⚠️ No image uploaded.", ""
        return

    thinking_buf = ""
    json_buf = ""

    for chunk_line in run_inference_stream(image):
        try:
            chunk = json.loads(chunk_line.strip())
            if chunk.get("thinking"):
                thinking_buf = chunk["thinking"]
            if chunk.get("json_output"):
                json_buf = chunk["json_output"]
            yield (
                thinking_buf or "⏳ Reasoning in progress…",
                json_buf or "⏳ Waiting for JSON output…"
            )
        except json.JSONDecodeError:
            continue

    yield (
        thinking_buf or "(No thinking trace in output)",
        json_buf or "(No JSON in output)"
    )


def build_gradio_ui():
    with gr.Blocks(title="Medical Document OCR") as demo:

        gr.HTML(f"<style>{CUSTOM_CSS}</style>")

        gr.HTML("""
        <div id="header-banner">
          <p id="header-title">🏥 Medical Document Understanding</p>
          <p id="header-sub">
            AI-powered OCR &nbsp;·&nbsp; Thinking model &nbsp;·&nbsp;
            Structured JSON extraction &nbsp;·&nbsp; OPD reports
          </p>
        </div>
        """)

        angle_state   = gr.State(0)
        orig_image    = gr.State(None)

        with gr.Row(equal_height=False):

            with gr.Column(scale=1, min_width=360):
                gr.HTML("""
                <div class="panel-label">
                  <span class="dot dot-blue"></span> Document Input
                </div>""")

                image_input = gr.Image(
                    type="pil",
                    label="",
                    height=380,
                    sources=["upload", "clipboard"],
                    show_label=False,
                    elem_id="main-image-input",
                )

                gr.HTML("""
                <div class="panel-label" style="margin-top:14px;">
                  <span class="dot dot-yellow"></span> Image Controls
                </div>""")

                with gr.Row():
                    btn_rot_cw  = gr.Button("↻ Rotate CW",  elem_classes=["toolbar"])
                    btn_rot_ccw = gr.Button("↺ Rotate CCW", elem_classes=["toolbar"])

                with gr.Row():
                    zoom_slider = gr.Slider(
                        minimum=50, maximum=300, value=100, step=10,
                        label="Zoom (%)",
                        interactive=True,
                    )

                zoom_js = """
                <script>
                (function() {
                  function applyZoom(val) {
                    const imgs = document.querySelectorAll('#main-image-input img');
                    imgs.forEach(img => {
                      img.style.transform = 'scale(' + (val / 100) + ')';
                      img.style.transformOrigin = 'center center';
                    });
                  }
                  setInterval(function() {
                    const slider = document.querySelector('input[type="range"]');
                    if (slider && slider._zoomBound !== true) {
                      slider._zoomBound = true;
                      slider.addEventListener('input', function() {
                        applyZoom(parseFloat(this.value));
                      });
                    }
                  }, 500);
                })();
                </script>
                """
                gr.HTML(zoom_js)

                with gr.Row():
                    btn_zoom_in  = gr.Button("🔍+ Zoom In",  elem_classes=["toolbar"])
                    btn_zoom_out = gr.Button("🔍− Zoom Out", elem_classes=["toolbar"])
                    btn_zoom_rst = gr.Button("⊙ Reset Zoom", elem_classes=["toolbar"])

                btn_zoom_in.click(
                    fn=lambda z: min(z + 20, 300),
                    inputs=[zoom_slider], outputs=[zoom_slider]
                )
                btn_zoom_out.click(
                    fn=lambda z: max(z - 20, 50),
                    inputs=[zoom_slider], outputs=[zoom_slider]
                )
                btn_zoom_rst.click(
                    fn=lambda: 100,
                    inputs=[], outputs=[zoom_slider]
                )

                zoom_apply_js = """
                function(zoom_val) {
                    const imgs = document.querySelectorAll('#main-image-input img');
                    imgs.forEach(img => {
                        img.style.transform = 'scale(' + (zoom_val / 100) + ')';
                        img.style.transformOrigin = 'center center';
                        img.style.transition = 'transform 0.2s ease';
                    });
                    return zoom_val;
                }
                """
                zoom_slider.change(
                    fn=lambda z: z,
                    inputs=[zoom_slider],
                    outputs=[zoom_slider],
                    js=zoom_apply_js,
                )

                btn_submit = gr.Button("⚡ Extract Medical Data", elem_id="submit-btn")
                btn_clear  = gr.Button("✕ Clear All",             elem_id="clear-btn")

                gr.HTML(f"""
                <div id="status-bar">
                  <div class="sb-item">Device <span class="sb-val">{DEVICE.upper()}</span></div>
                  <div class="sb-item">Max tokens <span class="sb-val">{MAX_NEW_TOKENS}</span></div>
                  <div class="sb-item">Min image <span class="sb-val">{MIN_IMAGE_SIZE}px</span></div>
                  <div class="sb-item">Model <span class="sb-val">thinking</span></div>
                </div>
                """)

            with gr.Column(scale=2, min_width=520):

                gr.HTML("""
                <div class="panel-label">
                  <span class="dot dot-yellow"></span> Model Reasoning  &nbsp;
                  <span style="font-size:0.68rem;color:#8b949e;">
                    (step-by-step analysis before extraction)
                  </span>
                </div>""")

                thinking_output = gr.Textbox(
                    label="",
                    lines=10,
                    max_lines=18,
                    placeholder="Thinking trace will appear here during generation…",
                    interactive=False,
                    show_label=False,
                    elem_id="thinking-box",
                )

                gr.HTML("""
                <div class="panel-label" style="margin-top:18px;">
                  <span class="dot dot-green"></span> Extracted JSON
                  <span style="font-size:0.68rem;color:#8b949e;">
                    (structured medical record)
                  </span>
                </div>""")

                json_output = gr.Textbox(
                    label="",
                    lines=18,
                    max_lines=40,
                    placeholder="Structured JSON will appear here…",
                    interactive=False,
                    show_label=False,
                    elem_id="json-box",
                )

        image_input.upload(
            fn=lambda img: img,
            inputs=[image_input],
            outputs=[orig_image],
        )
        image_input.change(
            fn=lambda img: img,
            inputs=[image_input],
            outputs=[orig_image],
        )

        btn_rot_cw.click(
            fn=rotate_image,
            inputs=[image_input, angle_state],
            outputs=[image_input, angle_state],
        )

        btn_rot_ccw.click(
            fn=rotate_image_ccw,
            inputs=[image_input, angle_state],
            outputs=[image_input, angle_state],
        )

        btn_submit.click(
            fn=gradio_predict_split,
            inputs=[image_input],
            outputs=[thinking_output, json_output],
        )

        btn_clear.click(
            fn=lambda: (None, "", "", 0, 0, None),
            inputs=[],
            outputs=[image_input, thinking_output, json_output, angle_state, zoom_slider, orig_image],
        )

    return demo


gradio_ui = build_gradio_ui()
app = gr.mount_gradio_app(app, gradio_ui, path="/gradio")


# ── STARTUP / SHUTDOWN ──
@app.on_event("startup")
async def startup_event():
    logger.info(f"🚀 API Startup | Device: {DEVICE} | Model: {MODEL_ID}")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("🛑 API Shutdown - Releasing resources...")
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
        gc.collect()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=7860,
        reload=False,
        log_level="info",
        timeout_keep_alive=300,
    )
