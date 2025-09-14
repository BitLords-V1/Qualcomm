from __future__ import annotations
from pathlib import Path
import io, json, platform
import numpy as np
import soundfile as sf
import onnxruntime as ort
from tokenizers import Tokenizer

# ---------- providers (macOS prefers CoreML, then MPS) ----------
def _pick_providers():
    avail = set(ort.get_available_providers())
    pref = []
    if platform.system() == "Darwin":
        if "CoreMLExecutionProvider" in avail:
            pref.append("CoreMLExecutionProvider")
        if "MPSExecutionProvider" in avail:
            pref.append("MPSExecutionProvider")
    pref.append("CPUExecutionProvider")
    # keep only those actually available
    return [p for p in pref if p in avail]

def _make_session(model_path: str) -> ort.InferenceSession:
    so = ort.SessionOptions()
    return ort.InferenceSession(model_path, providers=_pick_providers(), sess_options=so)

# ---------- simple resampler to 16 kHz mono ----------
def _to_mono_16k(wav: np.ndarray, sr: int) -> np.ndarray:
    if wav.ndim > 1:
        wav = wav.mean(axis=-1)
    wav = wav.astype(np.float32, copy=False)
    if sr == 16000:
        return wav
    n_out = max(1, int(round(len(wav) * 16000.0 / float(sr))))
    x_old = np.linspace(0.0, 1.0, num=len(wav), endpoint=False, dtype=np.float32)
    x_new = np.linspace(0.0, 1.0, num=n_out,  endpoint=False, dtype=np.float32)
    return np.interp(x_new, x_old, wav).astype(np.float32)

# ---------- Whisper log-mel (80 x T) -> [1,80,3000] ----------
def log_mel_spectrogram(
    wav: np.ndarray,
    sr: int = 16000,
    n_fft: int = 400,
    hop: int = 160,
    win: int = 400,  # kept for compatibility; window uses n_fft
    n_mels: int = 80,
    fmin: float = 0.0,
    fmax: float = 8000.0,
    max_frames: int = 3000,
) -> np.ndarray:
    if wav.ndim > 1:
        wav = wav.mean(-1)
    wav = wav.astype(np.float32, copy=False)

    pad = n_fft // 2
    x = np.pad(wav, (pad, pad), mode="reflect") if len(wav) > 0 else np.zeros(n_fft + 2*pad, dtype=np.float32)
    n_frames = max(1, 1 + (len(x) - n_fft) // hop)
    total_needed = (n_frames - 1) * hop + n_fft
    if len(x) < total_needed:
        x = np.pad(x, (0, total_needed - len(x)), mode="constant")
    frames = np.lib.stride_tricks.as_strided(
        x, shape=(n_frames, n_fft), strides=(x.strides[0] * hop, x.strides[0]), writeable=False
    )

    win_fn = np.hanning(n_fft).astype(np.float32)
    spec = np.fft.rfft(frames * win_fn, n=n_fft, axis=1)
    mag = (np.abs(spec) ** 2).astype(np.float32)

    def hz_to_mel(h): return 2595.0 * np.log10(1.0 + h/700.0)
    def mel_to_hz(m): return 700.0 * (10.0**(m/2595.0) - 1.0)

    n_freqs = n_fft // 2 + 1
    mels = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2, dtype=np.float32)
    hz = mel_to_hz(mels)
    bins = np.floor((n_fft + 1) * hz / sr).astype(int)
    bins = np.clip(bins, 0, n_fft // 2)

    fb = np.zeros((n_mels, n_freqs), dtype=np.float32)
    for i in range(n_mels):
        l, c, r = bins[i], bins[i+1], bins[i+2]
        if c > l: fb[i, l:c] = (np.arange(l, c) - l) / max(1, c - l)
        if r > c: fb[i, c:r] = (r - np.arange(c, r)) / max(1, r - c)

    mel = np.maximum(1e-10, mag @ fb.T)      # [frames, n_mels]
    mel = 10.0 * np.log10(mel)
    mel = np.clip((mel + 80.0) / 80.0, 0.0, 1.0)
    mel = mel.T                               # [n_mels, frames]

    if mel.shape[1] < max_frames:
        mel = np.pad(mel, ((0,0),(0, max_frames - mel.shape[1])), mode="constant")
    else:
        mel = mel[:, :max_frames]
    return mel[None, ...].astype(np.float32)  # [1,80,3000]


class WhisperPairTiny:
    """
    HfWhisperEncoder + HfWhisperDecoder ONNX via ONNX Runtime (CoreML/MPS/CPU).
    Greedy decode using local tokenizer.json. If config is missing, uses safe IDs.
    """
    def __init__(
        self,
        enc_path: Path = Path("../models/whisper/encoder.onnx/model.onnx"),
        dec_path: Path = Path("../models/whisper/decoder.onnx/model.onnx"),
        tok_path: Path | None = Path("../models/whisper/tokenizer.json"),
        cfg_path: Path | None = Path("../models/whisper/config.json"),
        gen_cfg_path: Path | None = Path("../models/whisper/generation_config.json"),
    ):
        # Sessions
        self.enc = _make_session(str(enc_path))
        self.dec = _make_session(str(dec_path))

        # Names
        self.enc_in  = self.enc.get_inputs()[0].name
        self.enc_out = self.enc.get_outputs()[0].name

        din = self.dec.get_inputs()
        self.dec_in_ids = next(i.name for i in din if "input_ids" in i.name or "decoder_input" in i.name)
        self.dec_in_enc = next(i.name for i in din if "encoder" in i.name or "hidden_state" in i.name)
        self.dec_out_logits = self.dec.get_outputs()[0].name

        # Optional extra inputs that some exports require
        self.dec_in_attn = next((i.name for i in din if "attention_mask" in i.name and "encoder" not in i.name), None)
        self.dec_in_pos  = next((i.name for i in din if "position_ids" in i.name), None)
        self.dec_in_enc_attn = next((i.name for i in din if "encoder_attention_mask" in i.name), None)

        # Tokenizer (required)
        tok_path = Path(tok_path) if tok_path is not None else None
        if not tok_path or not tok_path.exists():
            raise FileNotFoundError(
                "Missing tokenizer.json. Place it at models/whisper/tokenizer.json "
                "or pass tok_path=... (downloaded from the matching HF repo)."
            )
        self.tokenizer = Tokenizer.from_file(str(tok_path))

        # Config (optional; use safe defaults if absent)
        # Common Whisper tiny.en defaults:
        self.decoder_start = 50257  # <|startoftranscript|>
        self.eos           = 50256  # <|endoftext|>
        for cfg_file in [gen_cfg_path, cfg_path]:
            if cfg_file and Path(cfg_file).exists():
                try:
                    cfg = json.load(open(cfg_file, "r", encoding="utf-8"))
                    self.decoder_start = int(cfg.get("decoder_start_token_id", self.decoder_start))
                    self.eos = int(cfg.get("eos_token_id", self.eos))
                    break
                except Exception:
                    pass

        self.max_len = 224  # conservative cap

    def _encode(self, mel: np.ndarray) -> np.ndarray:
        return self.enc.run([self.enc_out], {self.enc_in: mel})[0]

    def _decode_greedy(self, enc_states: np.ndarray) -> str:
        ids = [self.decoder_start]
        for _ in range(self.max_len):
            feeds = {
                self.dec_in_ids: np.array(ids, dtype=np.int64)[None, :],
                self.dec_in_enc: enc_states,
            }
            # supply optional masks if the graph asks for them
            if self.dec_in_attn:
                feeds[self.dec_in_attn] = np.ones((1, len(ids)), dtype=np.int64)
            if self.dec_in_pos:
                feeds[self.dec_in_pos] = np.arange(len(ids), dtype=np.int64)[None, :]
            if self.dec_in_enc_attn:
                enc_len = enc_states.shape[1]
                feeds[self.dec_in_enc_attn] = np.ones((1, enc_len), dtype=np.int64)

            logits = self.dec.run([self.dec_out_logits], feeds)[0]  # [1, t, V]
            next_id = int(logits[0, -1].argmax(-1))
            if next_id == self.eos:
                break
            ids.append(next_id)

        return self.tokenizer.decode(ids[1:], skip_special_tokens=True).strip()

    def transcribe_bytes(self, wav_bytes: bytes) -> str:
        data, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32", always_2d=False)
        data = _to_mono_16k(data, sr)
        mel = log_mel_spectrogram(data, sr=16000)
        enc = self._encode(mel)
        return self._decode_greedy(enc)
