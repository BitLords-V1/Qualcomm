import io, base64
from flask import Flask, request, jsonify
from PIL import Image
from tts import TTS
from sst_whisper_pair import WhisperPairTiny

from ocr_pipeline import ocr_easy   # swap to a hybrid later if you add TrOCR

app = Flask(__name__)
tts = TTS(rate=180)
stt = WhisperPairTiny()   # uses models/whisper/{encoder.onnx,decoder.onnx}
_last_text = ""

@app.post("/ocr")
def ocr():
    """Accepts JSON { image_b64: 'data:image/...;base64,....' }"""
    global _last_text
    data = request.get_json(force=True)
    if not data or "image_b64" not in data:
        return jsonify({"error": "image_b64 missing"}), 400

    try:
        b64 = data["image_b64"].split(",", 1)[-1]
        img = Image.open(io.BytesIO(base64.b64decode(b64)))
    except Exception as e:
        return jsonify({"error": f"invalid image: {e}"}), 400

    text = ocr_easy(img)
    _last_text = text or ""
    return jsonify({"text": _last_text})

@app.post("/stt")
def stt_ep():
    """
    Accepts raw body audio/wav OR multipart file 'file' OR JSON { audio_b64: 'data:audio/wav;base64,...' }
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
            wav_bytes = base64.b64decode(b64.split(",", 1)[-1])

    if not wav_bytes:
        return jsonify({"error": "no audio provided"}), 400

    try:
        text = stt.transcribe_bytes(wav_bytes) or ""
        return jsonify({"text": text})
    except Exception as e:
        return jsonify({"error": f"stt failed: {e}"}), 500

@app.post("/command")
def command():
    """
    Accepts JSON { text: 'repeat|slower|faster|spell ...' }
    """
    global _last_text
    data = request.get_json(force=True)
    intent = (data.get("text", "") or "").strip().lower()

    if "repeat" in intent or "again" in intent:
        if _last_text:
            tts.say(_last_text)
        return jsonify({"ok": True, "intent": "repeat"})

    if "slower" in intent or "slow" in intent:
        tts.slower()
        return jsonify({"ok": True, "intent": "slower", "rate": tts.rate})

    if "faster" in intent or "speed" in intent:
        tts.faster()
        return jsonify({"ok": True, "intent": "faster", "rate": tts.rate})

    if "spell" in intent:
        if _last_text.strip():
            last = _last_text.strip().split()[-1]
            if last:
                tts.say(" ".join(list(last)))
        return jsonify({"ok": True, "intent": "spell"})

    return jsonify({"ok": True, "intent": "noop"})

if __name__ == "__main__":
    # Bind to localhost only; offline by design
    app.run(host="127.0.0.1", port=5005, debug=False)
