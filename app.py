# ============================================================
# Medical Document Understanding API (Production + Streaming)
# FastAPI + Gradio + Streaming + Robust Inference + Enhanced Preprocessing
# Updated: Auto-Orientation + Deskew + Contrast Enhancement + 1024px Min Image
# ============================================================

import os
import torch
import json
import re
import gc
import logging
import numpy as np
from threading import Thread
from PIL import Image, ImageOps, ImageEnhance

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse

from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    BitsAndBytesConfig,
    TextIteratorStreamer
)

import gradio as gr

# ============================================================
# CONFIG
# ============================================================
MODEL_ID = "Sebukpor/medical-document-understanding-v2"
HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    logging.warning("⚠️ HF_TOKEN not found in environment. Private repo access may fail.")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cuda.matmul.allow_tf32 = True

# Context window configuration
MAX_CONTEXT_TOKENS = 10048
MAX_NEW_TOKENS = 6048

# Image preprocessing configuration
MIN_IMAGE_SIZE = 1024

# Preprocessing behavior flags (tune for your dataset)
PREPROCESS_CONFIG = {
    "auto_rotate": True,           # Auto-detect 90/180/270° rotations
    "deskew": True,                # Apply minor skew correction via text-line analysis
    "deskew_threshold": 1.5,       # Min skew angle (degrees) to trigger correction
    "contrast_factor": 1.25,       # Contrast enhancement multiplier (1.0 = no change)
    "brightness_threshold": 128,   # Mean brightness below this triggers boost
    "brightness_factor": 1.1,      # Brightness boost multiplier
    "sharpness_factor": 1.0,       # Sharpness enhancement (1.0 = disabled, try 1.1 for fine text)
    "denoise": False,              # Apply median filter for noisy scans (slower)
}

# ============================================================
# PRECISION AUTO-SELECTION
# ============================================================
def get_precision_config():
    """Auto-select precision based on GPU capability for optimal performance."""
    if DEVICE == "cuda":
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            logger.info("🚀 Using BF16 precision (A100/L40/H100 detected)")
            return {"torch_dtype": torch.bfloat16, "quantization_config": None}
        else:
            logger.info("⚡ Using INT8 quantization (T4/V100 fallback)")
            return {
                "torch_dtype": torch.float16,
                "quantization_config": BitsAndBytesConfig(load_in_8bit=True)
            }
    else:
        logger.warning("🐌 Running on CPU - expect slower inference")
        return {"torch_dtype": torch.float32, "quantization_config": None}

precision_config = get_precision_config()

# ============================================================
# LOAD MODEL & PROCESSOR (Private Repo Auth)
# ============================================================
def load_components():
    """Load processor and model with authentication for private repos."""
    load_kwargs = {"token": HF_TOKEN, "trust_remote_code": True}

    logger.info("📥 Loading processor from private repo...")
    try:
        processor = AutoProcessor.from_pretrained(MODEL_ID, **load_kwargs)
    except Exception as e:
        logger.error(f"❌ Failed to load processor: {e}")
        raise RuntimeError(f"Processor load failed: {e}")

    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id
        logger.info("✅ Set pad_token_id to eos_token_id")

    logger.info("📥 Loading model with precision config...")
    try:
        model = AutoModelForImageTextToText.from_pretrained(
            MODEL_ID,
            device_map="auto",
            low_cpu_mem_usage=True,
            token=HF_TOKEN,
            trust_remote_code=True,
            **precision_config
        )
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        raise RuntimeError(f"Model load failed: {e}")

    model.eval()
    logger.info("✅ Model & processor loaded successfully")
    return processor, model

processor, model = load_components()

# ============================================================
# SYSTEM PROMPT (Enhanced for structured extraction)
# ============================================================
SYSTEM_PROMPT = (
    "### ROLE\n"
    "You are an expert Medical Transcriptionist and Data Extraction AI. "
    "Your goal is to digitize hand-written medical OPD (Outpatient Department) "
    "reports into a strictly structured JSON format.\n\n"
    "For the printed headers which has the demographic and patient details and bill details provide transcribe them as well as per the json field, do not leave them blank.\n\n"
    "### TASK\n"
    "Analyze the provided image. Distinguish between printed form headers and "
    "the doctor's handwritten notes. Extract all information into the JSON schema provided below.\n\n"
    "### EXTRACTION RULES\n"
    "1. Clinical Notes: For 'presenting_complaints', combine all handwritten text "
    "in that section into a single string.\n"
    "2. Medical Shorthand: Transcribe abbreviations exactly as written "
    "(e.g., 'C/o', 'H/o', 'RE/LE'). Do not expand them unless illegible.\n"
    "3. Prescriptions: Each medication must be a separate object in the 'prescription' list. "
    "Split drug name/dose, route, frequency, and duration.\n"
    "4. Booleans:\n"
    "   - consent_given = true only if a Yes checkbox is ticked or consent signature present.\n"
    "   - signature_present = true if doctor's signature/initials visible.\n"
    "5. Missing Data: Use empty string \"\" or null.\n"
    "6. Handwriting: Use \"[unclear]\" if text is illegible.\n"
    "7. Note: In some cases, doctors may record information in incorrect or inconsistent fields. Your task is to identify such instances and accurately reassign the information to the appropriate fields.\n\n"
    "8. Extract printed fields also (patient name, father name, bill no, etc).\n\n"
    "### JSON SCHEMA\n"
    "Return ONLY valid JSON with this structure:\n"
    "{\n"
    '  "patient_demographics": {\n'
    '    "patient_name": "",\n'
    '    "father_husband_name": "",\n'
    '    "age_sex": "",\n'
    '    "phone_number": "",\n'
    '    "address": "",\n'
    '    "appointment_date": "",\n'
    '    "uhid": "",\n'
    '    "bill_no": "",\n'
    '    "dept": ""\n'
    "  },\n"
    '  "clinical_notes": {\n'
    '    "presenting_complaints": "",\n'
    '    "history_of_allergies": "",\n'
    '    "examination_findings": "",\n'
    '    "investigations_advised": "",\n'
    '    "result_of_investigations": "",\n'
    '    "provisional_diagnosis": "",\n'
    '    "nutritional_needs": ""\n'
    "  },\n"
    '  "procedures": {\n'
    '    "consent_given": false,\n'
    '    "procedure_note": ""\n'
    "  },\n"
    '  "prescription": [\n'
    "    {\n"
    '      "drug_dose": "",\n'
    '      "route": "",\n'
    '      "frequency": "",\n'
    '      "duration": ""\n'
    "    }\n"
    "  ],\n"
    '  "follow_up": {\n'
    '    "date": "",\n'
    '    "urgent_care_instructions": ""\n'
    "  },\n"
    '  "doctor_details": {\n'
    '    "signature_present": false,\n'
    '    "name_stamp": ""\n'
    "  }\n"
    "}\n\n"
    "Return only valid JSON. No explanations, no markdown."
)

# ============================================================
# IMAGE PREPROCESSING (Enhanced: Orientation + Deskew + Contrast)
# ============================================================
def preprocess_image(image: Image.Image) -> Image.Image:
    """
    Preprocess image for optimal model input:
    - Handle EXIF orientation + detect major rotations (90/180/270°)
    - Deskew minor tilts using text-line analysis
    - Adaptive contrast/brightness enhancement for handwritten text
    - Resize to minimum 1024px while maintaining aspect ratio
    - Optional: denoising and sharpness for OCR boost
    """
    try:
        # ── STEP 1: Handle EXIF orientation ──────────────────
        image = ImageOps.exif_transpose(image)
        if image.mode != "RGB":
            image = image.convert("RGB")

        # ── STEP 2: Detect and correct major rotations ───────
        if PREPROCESS_CONFIG.get("auto_rotate", True):
            width, height = image.size
            aspect_ratio = height / width if width > 0 else 1
            # Medical forms are typically portrait; if landscape, rotate 90°
            if aspect_ratio < 0.7:
                logger.debug("🔄 Detected landscape orientation, rotating 90°")
                image = image.rotate(90, expand=True, resample=Image.Resampling.BICUBIC)

        # ── STEP 3: Deskew minor tilts using text-line analysis ──
        if PREPROCESS_CONFIG.get("deskew", True):
            try:
                import cv2
                from deskew import determine_skew
                
                img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
                angle = determine_skew(gray)
                threshold = PREPROCESS_CONFIG.get("deskew_threshold", 1.5)
                
                if abs(angle) > threshold:
                    logger.debug(f"📐 Detected skew: {angle:.2f}°, correcting...")
                    image = image.rotate(
                        angle,
                        resample=Image.Resampling.BICUBIC,
                        expand=True,
                        fillcolor=(255, 255, 255)
                    )
            except ImportError:
                logger.debug("⚠️ deskew/OpenCV not available, skipping deskew step")
            except Exception as e:
                logger.warning(f"⚠️ Deskew failed: {e}, continuing with original orientation")

        # ── STEP 4: Resize to minimum dimension ───────────────
        if min(image.size) < MIN_IMAGE_SIZE:
            scale = MIN_IMAGE_SIZE / min(image.size)
            new_size = (
                max(int(image.width * scale), MIN_IMAGE_SIZE),
                max(int(image.height * scale), MIN_IMAGE_SIZE)
            )
            image = image.resize(new_size, Image.Resampling.LANCZOS)
            logger.debug(f"🖼️ Resized image to {new_size}")

        # ── STEP 5: Adaptive contrast/brightness enhancement ──
        contrast_factor = PREPROCESS_CONFIG.get("contrast_factor", 1.25)
        if contrast_factor != 1.0:
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(contrast_factor)
        
        # Brightness boost for dark images
        brightness_threshold = PREPROCESS_CONFIG.get("brightness_threshold", 128)
        brightness_factor = PREPROCESS_CONFIG.get("brightness_factor", 1.1)
        img_array = np.array(image)
        mean_brightness = np.mean(img_array)
        if mean_brightness < brightness_threshold and brightness_factor != 1.0:
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(brightness_factor)
            logger.debug(f"💡 Brightness boosted (mean: {mean_brightness:.1f})")

        # Optional sharpness enhancement
        sharpness_factor = PREPROCESS_CONFIG.get("sharpness_factor", 1.0)
        if sharpness_factor != 1.0:
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(sharpness_factor)

        # Optional denoising for grainy scans
        if PREPROCESS_CONFIG.get("denoise", False):
            image = image.filter(Image.Filter.MedianFilter(size=3))

        return image

    except Exception as e:
        logger.warning(f"⚠️ Image preprocessing failed: {e}. Using fallback processing.")
        # Fallback: ensure basic EXIF handling and RGB format
        try:
            image = ImageOps.exif_transpose(image)
            return image.convert("RGB") if image.mode != "RGB" else image
        except:
            return image

# ============================================================
# BUILD MESSAGES FOR CHAT TEMPLATE
# ============================================================
def build_messages(pil_image: Image.Image) -> list:
    """Construct chat-formatted messages for the vision-language model."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": pil_image},
                {"type": "text", "text": "Extract all medical information into structured JSON format following the schema rules."},
            ],
        },
    ]

# ============================================================
# ROBUST JSON PARSER
# ============================================================
def parse_json_robust(text: str) -> dict:
    """Extract and parse JSON from model output with multiple fallback strategies."""
    if not text:
        return {"error": "Empty response", "raw_output": ""}

    text = re.sub(r"```json\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"```\s*$", "", text, flags=re.IGNORECASE)
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    json_matches = re.findall(r'\{(?:[^{}]|(?R))*\}', text, re.DOTALL)
    if not json_matches:
        json_matches = re.findall(r'\{.*?\}', text, re.DOTALL)
    
    for match in reversed(json_matches):
        try:
            return json.loads(match)
        except json.JSONDecodeError:
            continue

    try:
        fixed_text = re.sub(r",\s*}", "}", text)
        fixed_text = re.sub(r",\s*]", "]", fixed_text)
        fixed_text = re.sub(r"'", '"', fixed_text)
        return json.loads(fixed_text)
    except:
        pass

    logger.warning(f"⚠️ JSON parsing failed. Raw output preview: {text[:200]}...")
    return {
        "error": "JSON parsing failed",
        "raw_output": text,
        "suggestion": "Check model output format or retry with clearer image"
    }

# ============================================================
# STANDARD INFERENCE (Non-streaming)
# ============================================================
@torch.no_grad()
def run_inference(image: Image.Image) -> dict:
    """Run standard inference with memory management fallback."""
    try:
        image = preprocess_image(image)
        messages = build_messages(image)
        prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[prompt_text], images=[image], return_tensors="pt", padding=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        outputs = None
        for max_tokens in [MAX_NEW_TOKENS, MAX_NEW_TOKENS // 2, MAX_NEW_TOKENS // 4]:
            try:
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=False,
                    temperature=0.1,
                    repetition_penalty=1.15,
                )
                break
            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                if "out of memory" in str(e).lower():
                    logger.warning(f"⚠️ OOM with {max_tokens} tokens, retrying with fewer...")
                    torch.cuda.empty_cache()
                    gc.collect()
                    continue
                raise

        if outputs is None:
            raise RuntimeError("Failed to generate output after memory fallback attempts")

        decoded = processor.batch_decode(outputs, skip_special_tokens=True)[0]
        return parse_json_robust(decoded)

    except Exception as e:
        logger.error(f"❌ Inference error: {e}", exc_info=True)
        return {"error": str(e), "raw_output": None}

# ============================================================
# STREAMING INFERENCE
# ============================================================
@torch.no_grad()
def run_inference_stream(image: Image.Image):
    """Run streaming inference with TextIteratorStreamer."""
    try:
        image = preprocess_image(image)
        messages = build_messages(image)
        prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[prompt_text], images=[image], return_tensors="pt", padding=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        streamer = TextIteratorStreamer(
            processor.tokenizer,
            skip_special_tokens=True,
            skip_prompt=True,
            timeout=30.0
        )

        generation_kwargs = dict(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            temperature=0.1,
            repetition_penalty=1.15,
            streamer=streamer,
            pad_token_id=processor.tokenizer.pad_token_id
        )

        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()

        partial_text = ""
        for new_token in streamer:
            partial_text += new_token
            yield partial_text

        thread.join()

    except Exception as e:
        logger.error(f"❌ Streaming inference error: {e}", exc_info=True)
        yield f"Error: {str(e)}"

# ============================================================
# FASTAPI APPLICATION
# ============================================================
app = FastAPI(
    title="Medical Document Understanding API",
    description="Production API for extracting structured JSON from medical OPD documents",
    version="1.2.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

@app.get("/", tags=["Health"])
async def health_check():
    return {
        "status": "healthy",
        "model": MODEL_ID,
        "device": DEVICE,
        "context_window": MAX_CONTEXT_TOKENS,
        "min_image_size": MIN_IMAGE_SIZE,
        "preprocessing": PREPROCESS_CONFIG
    }

@app.post("/predict", tags=["Inference"], response_class=JSONResponse)
async def predict(file: UploadFile = File(...)):
    """Standard inference endpoint."""
    try:
        if not file.content_type or not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")
        image = Image.open(file.file)
        result = run_inference(image)
        return JSONResponse(content=result)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ /predict error: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"error": "Inference failed", "details": str(e)})
    finally:
        file.file.close()

@app.post("/predict_stream", tags=["Inference"], response_class=StreamingResponse)
async def predict_stream(file: UploadFile = File(...)):
    """Streaming inference endpoint."""
    try:
        if not file.content_type or not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")
        image = Image.open(file.file).convert("RGB")

        def generator():
            try:
                for chunk in run_inference_stream(image):
                    yield chunk
            except Exception as e:
                logger.error(f"❌ Stream generator error: {e}")
                yield f"\n\n[Stream Error: {str(e)}]"

        return StreamingResponse(
            generator(),
            media_type="text/plain",
            headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"}
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ /predict_stream error: {e}", exc_info=True)
        return StreamingResponse(iter([f"Error: {str(e)}"]), media_type="text/plain", status_code=500)
    finally:
        file.file.close()

# Debug endpoint for preprocessing visualization
@app.post("/debug/preprocess", tags=["Debug"])
async def debug_preprocess(file: UploadFile = File(...)):
    """Return original vs preprocessed image for validation."""
    try:
        from io import BytesIO
        original = Image.open(file.file).convert("RGB")
        processed = preprocess_image(original.copy())
        
        total_width = original.width + processed.width
        max_height = max(original.height, processed.height)
        comparison = Image.new("RGB", (total_width, max_height), color="white")
        comparison.paste(original, (0, 0))
        comparison.paste(processed, (original.width, 0))
        
        buf = BytesIO()
        comparison.save(buf, format="PNG")
        buf.seek(0)
        return StreamingResponse(buf, media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Preprocess debug failed: {e}")

# ============================================================
# GRADIO INTERFACE
# ============================================================
def gradio_predict(image):
    """Gradio wrapper for streaming inference."""
    if image is None:
        return "Please upload an image first."
    output_text = ""
    for chunk in run_inference_stream(image):
        output_text = chunk
        yield output_text
    return output_text

gradio_app = gr.Interface(
    fn=gradio_predict,
    inputs=gr.Image(type="pil", label="Upload Medical Document", height=400, sources=["upload", "clipboard"]),
    outputs=gr.Textbox(label="Extracted JSON (Streaming)", lines=25, max_lines=50, show_copy_button=True),
    title="🏥 Medical Document Understanding",
    description=(
        f"Upload a handwritten or printed medical OPD report to extract structured JSON. "
        f"Model: {MODEL_ID} | Min Image: {MIN_IMAGE_SIZE}px | Preprocessing: Auto-rotate + Deskew + Contrast"
    ),
    cache_examples=False,
    allow_flagging="never",
    theme=gr.themes.Soft(primary_hue="blue", secondary_hue="slate"),
)

app = gr.mount_gradio_app(app, gradio_app, path="/gradio")

# ============================================================
# STARTUP/SHUTDOWN EVENTS
# ============================================================
@app.on_event("startup")
async def startup_event():
    logger.info(f"🚀 API Startup | Device: {DEVICE} | Model: {MODEL_ID}")
    logger.info(f"⚙️ Config: Context={MAX_CONTEXT_TOKENS}, MinImgSize={MIN_IMAGE_SIZE}px")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("🛑 API Shutdown - Releasing resources...")
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    logger.info("✅ Cleanup complete")

# ============================================================
# RUN COMMAND
# ============================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=7860,
        reload=False,
        log_level="info",
        timeout_keep_alive=300
    )