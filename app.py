"""
Flask backend API for J-Nita v5.0.

Endpoints:
  GET  /                      Serve frontend
  GET  /api/health            Liveness check
  POST /api/condition         Pre-process an uploaded image
  POST /api/ocr               Run OCR on a conditioned image
  GET  /api/config            Read config.json
  POST /api/export/pdf        Export markdown as PDF
  POST /api/export/docx       Export markdown as DOCX
"""

import os
import io
import json
import base64
import logging
import tempfile

import cv2
import numpy as np
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from PIL import Image, ImageOps

from image_conditioner import ImageConditioner
from orientation_validator import detect_gibberish
from gradio_client import Client, handle_file


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "config.json")

MAX_UPLOAD_BYTES = 30 * 1024 * 1024     # Reject requests larger than 30 MB
MAX_INPUT_DIMENSION = 1800              # Cap input pixels before processing
MAX_OUTPUT_DIMENSION = 2000             # Cap output PNG dimensions
STAGE_PREVIEW_DIMENSION = 800           # Stage previews are small thumbnails
PNG_COMPRESSION = 6                     # 0-9; 6 is a good speed/size tradeoff
JPEG_QUALITY = 75                       # Stage previews

OCR_HF_SPACE = "prithivMLmods/Multimodal-OCR3"
DEFAULT_OCR_MODEL = "Nanonets-OCR2-3B"
OCR_RETRY_KEYWORDS = ("timeout", "gpu", "connection", "queue", "retry", "503", "504")


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES
CORS(app, resources={r"/api/*": {"origins": "*"}})

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

_ocr_client = None  # Lazily initialized by _get_ocr_client()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_ocr_client():
    global _ocr_client
    if _ocr_client is None:
        _ocr_client = Client(OCR_HF_SPACE)
        logger.info("OCR client initialized for %s", OCR_HF_SPACE)
    return _ocr_client


def _load_config():
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning("Could not load config.json: %s", e)
        return {}


def _config_value(cfg, section, key, default=None):
    """Read cfg[section][key]['value'], returning default if any level is missing."""
    return cfg.get(section, {}).get(key, {}).get("value", default)


def _decode_image_payload(data_url_or_b64):
    """Decode a base64 image (with or without data URL prefix) into a BGR ndarray.

    Honors EXIF orientation and pre-resizes to MAX_INPUT_DIMENSION so the
    downstream pipeline stays fast on Render's small instances.
    """
    if "," in data_url_or_b64:
        data_url_or_b64 = data_url_or_b64.split(",", 1)[1]
    raw = base64.b64decode(data_url_or_b64, validate=False)

    pil = Image.open(io.BytesIO(raw))
    pil = ImageOps.exif_transpose(pil).convert("RGB")
    if max(pil.size) > MAX_INPUT_DIMENSION:
        pil.thumbnail((MAX_INPUT_DIMENSION, MAX_INPUT_DIMENSION), Image.Resampling.LANCZOS)
    arr = np.asarray(pil)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def _encode_png_b64(image, max_dim=MAX_OUTPUT_DIMENSION, compression=PNG_COMPRESSION):
    h, w = image.shape[:2]
    if max(h, w) > max_dim:
        s = max_dim / max(h, w)
        image = cv2.resize(image, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".png", image, [cv2.IMWRITE_PNG_COMPRESSION, compression])
    if not ok:
        raise RuntimeError("PNG encoding failed")
    return f"data:image/png;base64,{base64.b64encode(buf).decode('ascii')}"


def _encode_jpeg_b64(image, max_dim=STAGE_PREVIEW_DIMENSION, quality=JPEG_QUALITY):
    h, w = image.shape[:2]
    if max(h, w) > max_dim:
        s = max_dim / max(h, w)
        image = cv2.resize(image, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return f"data:image/jpeg;base64,{base64.b64encode(buf).decode('ascii')}"


def _build_conditioner(overrides):
    """Create an ImageConditioner from config.json with per-request overrides."""
    cfg = _load_config()
    ic = cfg.get("image_conditioning", {})
    for key, value in (overrides or {}).items():
        if key in ic and isinstance(ic[key], dict) and "value" in ic[key]:
            ic[key]["value"] = value

    def v(k, default=None):
        return ic.get(k, {}).get("value", default)

    morph_kernel = v("morph_kernel_size", [2, 2])
    tmp = tempfile.gettempdir()  # ImageConditioner needs writable folders even when unused
    return ImageConditioner(
        input_folder=tmp,
        output_folder=tmp,
        strength=v("strength", 10),
        adaptive_block_size=v("adaptive_block_size"),
        adaptive_C=v("adaptive_C"),
        morph_kernel_size=tuple(morph_kernel) if morph_kernel else None,
        morph_iterations=v("morph_iterations"),
        target_width=v("target_width"),
        target_height=v("target_height"),
        canny_threshold1=v("canny_threshold1"),
        canny_threshold2=v("canny_threshold2"),
        min_contour_area=v("min_contour_area"),
        denoise_ksize=v("denoise_ksize"),
        png_compression=v("png_compression", PNG_COMPRESSION),
        min_scale_for_zero=v("min_scale_for_zero"),
        debug_mode=False,
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return send_file(os.path.join(BASE_DIR, "index.html"))


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/api/condition", methods=["POST"])
def condition_endpoint():
    data = request.get_json(silent=True) or {}
    image_data = data.get("image")
    if not image_data:
        return jsonify({"error": "No image provided", "success": False}), 400

    try:
        image_array = _decode_image_payload(image_data)
    except Exception as e:
        logger.warning("Could not decode image: %s", e)
        return jsonify({"error": f"Could not decode image: {e}", "success": False}), 400

    return_stages = bool(data.get("return_stages", False))
    overrides = data.get("config") or {}

    try:
        conditioner = _build_conditioner(overrides)
        if return_stages:
            output, stages = conditioner.condition_image_array(
                image_array, filename="upload", return_stages=True
            )
            return jsonify({
                "conditioned_image": _encode_png_b64(output),
                "stages": {name: _encode_jpeg_b64(img) for name, img in stages.items()},
                "success": True,
            })
        output = conditioner.condition_image_array(image_array, filename="upload")
        return jsonify({
            "conditioned_image": _encode_png_b64(output),
            "success": True,
        })
    except Exception as e:
        logger.exception("Conditioning failed")
        return jsonify({"error": str(e), "success": False}), 500


@app.route("/api/ocr", methods=["POST"])
def ocr_endpoint():
    data = request.get_json(silent=True) or {}
    image_data = data.get("image")
    if not image_data:
        return jsonify({"error": "No image provided", "success": False}), 400

    cfg = _load_config()
    handwriting_mode = bool(data.get("handwriting_mode", False))

    if handwriting_mode:
        model = _config_value(cfg, "ocr_workflow", "handwriting_model", "olmOCR-2-7B-1025")
        prompt = _config_value(
            cfg, "ocr_workflow", "handwriting_prompt",
            "This image contains handwritten text. Perform OCR and output well-formatted Markdown.",
        )
    else:
        model = data.get("model") or _config_value(cfg, "ocr_workflow", "ocr_model", DEFAULT_OCR_MODEL)
        prompt = "Perform OCR on the image."

    enable_fallback = bool(data.get("enable_fallback", True))
    fallback_models = data.get("fallback_models")
    if fallback_models is None:
        fallback_models = _config_value(cfg, "ocr_workflow", "fallback_models", []) or []

    try:
        b64 = image_data.split(",", 1)[1] if "," in image_data else image_data
        image_bytes = base64.b64decode(b64, validate=False)
    except Exception as e:
        return jsonify({"error": f"Invalid image: {e}", "success": False}), 400

    fd, tmp_path = tempfile.mkstemp(suffix=".png")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(image_bytes)

        client = _get_ocr_client()
        models_to_try = [model]
        if enable_fallback:
            models_to_try.extend(m for m in fallback_models if m != model)

        raw_output = ""
        markdown_output = ""
        model_used = None
        last_error = ""
        is_gibberish = False
        gibberish_score = 0.0
        gibberish_reason = ""

        for model_name in models_to_try:
            try:
                logger.info("Trying OCR model %s (handwriting=%s)", model_name, handwriting_mode)
                result = client.predict(
                    model_name, prompt, handle_file(tmp_path),
                    2048, 0.7, 0.9, 50, 1.1, 60,
                    api_name="/run_ocr",
                )
                raw_output = str(result[0]) if isinstance(result, (list, tuple)) and result else ""
                markdown_output = (
                    str(result[1]) if isinstance(result, (list, tuple)) and len(result) > 1
                    else raw_output
                )
                model_used = model_name

                if raw_output:
                    try:
                        is_gibberish, gibberish_score, gibberish_reason = detect_gibberish(raw_output)
                    except Exception as e:
                        logger.warning("Gibberish detection failed: %s", e)
                        is_gibberish, gibberish_score, gibberish_reason = False, 0.0, ""

                    if is_gibberish and enable_fallback and model_name != models_to_try[-1]:
                        last_error = f"Gibberish detected: {gibberish_reason}"
                        logger.warning("Gibberish from %s, trying next model", model_name)
                        continue

                logger.info("OCR success with %s", model_name)
                break

            except Exception as e:
                last_error = str(e)
                logger.warning("OCR failed with %s: %s", model_name, last_error)
                if not any(k in last_error.lower() for k in OCR_RETRY_KEYWORDS):
                    break

        if not raw_output and not markdown_output:
            return jsonify({
                "error": f"All models failed. Last error: {last_error}",
                "success": False,
            }), 502

        return jsonify({
            "raw_text": raw_output,
            "markdown_text": markdown_output,
            "model_used": model_used,
            "is_gibberish": is_gibberish,
            "gibberish_score": gibberish_score,
            "gibberish_reason": gibberish_reason,
            "success": True,
        })

    except Exception as e:
        logger.exception("OCR endpoint error")
        return jsonify({"error": str(e), "success": False}), 500
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


@app.route("/api/config", methods=["GET"])
def get_config():
    return jsonify(_load_config())


@app.route("/api/export/pdf", methods=["POST"])
def export_pdf():
    try:
        from fpdf import FPDF
    except ImportError:
        return jsonify({"error": "fpdf2 not installed"}), 500

    data = request.get_json(silent=True) or {}
    text = data.get("markdown_text", "")
    if not text:
        return jsonify({"error": "No text provided"}), 400

    # Core FPDF fonts only support latin-1; replace anything else so we never crash
    safe_text = text.encode("latin-1", errors="replace").decode("latin-1")

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=12)
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.multi_cell(0, 8, safe_text)

    buf = io.BytesIO(bytes(pdf.output()))
    return send_file(
        buf,
        mimetype="application/pdf",
        as_attachment=True,
        download_name=data.get("filename") or "ocr_result.pdf",
    )


@app.route("/api/export/docx", methods=["POST"])
def export_docx():
    try:
        from docx import Document
    except ImportError:
        return jsonify({"error": "python-docx not installed"}), 500

    data = request.get_json(silent=True) or {}
    text = data.get("markdown_text", "")
    if not text:
        return jsonify({"error": "No text provided"}), 400

    doc = Document()
    for paragraph in text.split("\n\n"):
        doc.add_paragraph(paragraph)

    buf = io.BytesIO()
    doc.save(buf)
    buf.seek(0)
    return send_file(
        buf,
        mimetype="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        as_attachment=True,
        download_name=data.get("filename") or "ocr_result.docx",
    )


@app.errorhandler(413)
def request_too_large(_e):
    return jsonify({
        "error": f"Upload exceeds {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit",
        "success": False,
    }), 413
