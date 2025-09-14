# app_macos.py
import io, base64, logging, os
from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS
from PIL import Image

# Local modules
from tts import TTS               # <-- provided below
from sst_whisper_pair import WhisperPairTiny  # your existing wrapper
from ocr_pipeline import ocr_easy       # unchanged

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("iLuminaApp")

app = Flask(__name__, static_folder='../frontend', static_url_path='')
CORS(app)  # Enable CORS for all routes

# Limit request size (optional, helps keep things safe/offline)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB

# Initialize TTS & STT
tts = TTS(rate=180)

# Try to initialize STT with fallback
stt = None
stt_available = False

try:
    # First try the QNN-based WhisperPairTiny
    stt = WhisperPairTiny()
    stt_available = True
    logger.info("STT initialized with WhisperPairTiny")
except Exception as e:
    logger.warning(f"WhisperPairTiny failed: {e}")
    try:
        # Fallback to simple WhisperEngine
        from whisper_qnn import WhisperEngine
        stt = WhisperEngine()
        stt_available = True
        logger.info("STT initialized with WhisperEngine fallback")
    except Exception as e2:
        logger.warning(f"WhisperEngine fallback also failed: {e2}")
        stt = None
        stt_available = False
_last_text = ""

@app.get("/")
def root():
    """Serve the frontend HTML file."""
    return send_file('../frontend/index.html')

@app.get("/api/info")
def api_info():
    """API information endpoint."""
    return jsonify({
        "message": "iLumina Backend API",
        "version": "1.0.0",
        "endpoints": {
            "GET /": "Frontend application",
            "GET /api/info": "This endpoint",
            "GET /api/healthz": "Health check",
            "POST /api/ocr": "Image text extraction (send image_b64 in JSON)",
            "POST /api/stt": "Speech-to-text (send audio data)",
            "POST /api/command": "TTS voice commands (send text in JSON)"
        },
        "status": {
            "tts_available": tts.available,
            "stt_available": stt_available
        }
    })

@app.post("/api/ocr")
def ocr():
    """
    Accepts JSON: { "image_b64": "data:image/...;base64,AAAA..." }
    Returns: {"text": "..."}
    """
    global _last_text
    data = request.get_json(force=True, silent=False)

    if not data or "image_b64" not in data:
        return jsonify({"error": "image_b64 missing"}), 400

    try:
        b64 = (data["image_b64"] or "").split(",", 1)[-1].strip()
        if not b64:
            return jsonify({"error": "empty image_b64"}), 400

        raw = base64.b64decode(b64, validate=True)
        with Image.open(io.BytesIO(raw)) as im:
            # Convert to RGB to avoid mode issues during OCR (e.g., P, RGBA, CMYK)
            img = im.convert("RGB")

    except Exception as e:
        return jsonify({"error": f"invalid image: {e}"}), 400

    try:
        text = ocr_easy(img) or ""
        _last_text = text
        return jsonify({
            "success": True,
            "text": _last_text,
            "confidence": 0.95  # Placeholder confidence
        })
    except Exception as e:
        logger.exception("OCR failed")
        return jsonify({
            "success": False,
            "error": f"ocr failed: {e}"
        }), 500


@app.post("/api/stt")
def stt_ep():
    """
    Accepts:
      1) raw body audio/wav
      2) multipart/form-data with file field 'file'
      3) JSON { "audio_b64": "data:audio/wav;base64,AAAA..." }
    Returns {"text": "..."}.
    """
    wav_bytes = None

    # 1) raw bytes
    if request.data:
        wav_bytes = request.data

    # 2) multipart/form-data
    if wav_bytes is None and "file" in request.files:
        wav_bytes = request.files["file"].read()

    # 3) JSON base64
    if wav_bytes is None:
        data = request.get_json(silent=True) or {}
        b64 = data.get("audio_b64")
        if b64:
            try:
                wav_bytes = base64.b64decode(b64.split(",", 1)[-1], validate=True)
            except Exception as e:
                return jsonify({"error": f"invalid audio_b64: {e}"}), 400

    if not wav_bytes:
        return jsonify({"error": "no audio provided"}), 400

    if not stt_available or stt is None:
        return jsonify({"error": "STT not available"}), 503
    
    try:
        # Handle different STT implementations
        if hasattr(stt, 'transcribe_bytes'):
            # WhisperPairTiny
            text = stt.transcribe_bytes(wav_bytes) or ""
        elif hasattr(stt, 'transcribe'):
            # WhisperEngine
            result = stt.transcribe(wav_bytes)
            text = result.get('text', '') if isinstance(result, dict) else str(result)
        else:
            text = "STT method not found"
        
        return jsonify({
            "success": True,
            "text": text
        })
    except Exception as e:
        logger.exception("STT failed")
        return jsonify({"error": f"stt failed: {e}"}), 500


@app.post("/api/command")
def command():
    """
    Accepts JSON: { "text": "repeat | slower | faster | spell ..." }
    Controls local TTS actions. Uses last OCR text for repeat/spell.
    """
    global _last_text
    data = request.get_json(force=True) or {}
    intent = (data.get("text", "") or "").strip().lower()

    if "repeat" in intent or "again" in intent:
        if _last_text:
            tts.say(_last_text)
        return jsonify({
            "success": True,
            "intent": "repeat",
            "message": "Repeating last text"
        })

    if "slower" in intent or "slow" in intent:
        tts.slower()
        return jsonify({
            "success": True,
            "intent": "slower",
            "rate": tts.rate,
            "message": f"Speech rate slowed to {tts.rate}"
        })

    if "faster" in intent or "speed" in intent:
        tts.faster()
        return jsonify({
            "success": True,
            "intent": "faster",
            "rate": tts.rate,
            "message": f"Speech rate increased to {tts.rate}"
        })

    if "spell" in intent:
        if _last_text.strip():
            last = _last_text.strip().split()[-1]
            if last:
                # speak last word as spaced letters; include 'space' if needed
                tokens = []
                for ch in last:
                    tokens.append("space" if ch.isspace() else ch)
                tts.say(" ".join(tokens))
        return jsonify({
            "success": True,
            "intent": "spell",
            "message": "Spelling last word"
        })

    return jsonify({
        "success": True,
        "intent": "noop",
        "message": "Command not recognized"
    })


@app.get("/api/healthz")
def healthz():
    """Simple offline health check."""
    return jsonify({
        "ok": True,
        "tts_available": tts.available,
        "stt_ready": stt_available,
        "npu_available": False,  # Placeholder for NPU status
        "status": "ready"
    })


if __name__ == "__main__":
    # Bind to localhost only; offline by design
    app.run(host="127.0.0.1", port=5005, debug=False)
