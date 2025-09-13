from __future__ import annotations
from pathlib import Path
import io, json
import numpy as np
import soundfile as sf
from tokenizers import Tokenizer
from .qnn_runtime import make_qnn_session

# ---------- audio -> log-mel (Whisper settings) ----------
def log_mel_spectrogram(wav: np.ndarray, sr: int = 16000,
                        n_fft=400, hop=160, win=400, n_mels=80,
                        fmin=0, fmax=8000, max_frames=3000) -> np.ndarray:
    # pad/center
    if wav.ndim > 1: wav = wav.mean(-1)
    wav = wav.astype(np.float32)
    # frame
    pad = n_fft // 2
    x = np.pad(wav, (pad, pad), mode="reflect")
    n_frames = 1 + (len(x) - n_fft) // hop
    frames = np.lib.stride_tricks.as_strided(
        x,
        shape=(n_frames, n_fft),
        strides=(x.strides[0]*hop, x.strides[0]),
        writeable=False
    )
    # window + rfft
    win_fn = np.hanning(n_fft).astype(np.float32)
    spec = np.fft.rfft(frames * win_fn, n=n_fft, axis=1)
    mag = (np.abs(spec) ** 2).astype(np.float32)

    # mel filterbank
    def hz_to_mel(h): return 2595.0 * np.log10(1.0 + h/700.0)
    def mel_to_hz(m): return 700.0 * (10.0**(m/2595.0) - 1.0)

    n_freqs = n_fft // 2 + 1
    fft_freqs = np.linspace(0, sr/2, n_freqs)
    mels = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
    hz = mel_to_hz(mels)
    bins = np.floor((n_fft + 1) * hz / sr).astype(int)
    fb = np.zeros((n_mels, n_freqs), dtype=np.float32)
    for i in range(n_mels):
        l, c, r = bins[i], bins[i+1], bins[i+2]
        if c > l:
            fb[i, l:c] = (np.arange(l, c) - l) / max(1, (c - l))
        if r > c:
            fb[i, c:r] = (r - np.arange(c, r)) / max(1, (r - c))

    mel = np.maximum(1e-10, mag @ fb.T)  # [frames, n_mels]
    mel = 10.0 * np.log10(mel)           # log10
    mel = np.clip((mel + 80.0) / 80.0, 0.0, 1.0)  # normalize to ~[0,1] like Whisper
    mel = mel.T  # [n_mels, frames]

    # pad/trim to 30s (=3000 frames at 100 fps)
    if mel.shape[1] < max_frames:
        mel = np.pad(mel, ((0,0), (0, max_frames - mel.shape[1])), mode="constant")
    else:
        mel = mel[:, :max_frames]
    return mel[None, ...].astype(np.float32)  # [1, 80, 3000]


class WhisperPairTiny:
    """
    Runs HfWhisperEncoder + HfWhisperDecoder ONNX with ONNX Runtime QNN EP.
    Greedy decode using HF tokenizer.json (offline).
    """
    def __init__(self,
                 enc_path: Path = Path("models/whisper/encoder.onnx"),
                 dec_path: Path = Path("models/whisper/decoder.onnx"),
                 tok_path: Path = Path("models/whisper/tokenizer.json"),
                 cfg_path: Path = Path("models/whisper/config.json")):

        self.enc = make_qnn_session(str(enc_path))
        self.dec = make_qnn_session(str(dec_path))

        # Infer common input/output names
        self.enc_in = self.enc.get_inputs()[0].name          # "input_features"
        self.enc_out = self.enc.get_outputs()[0].name        # "last_hidden_state" (1, T, D) or similar

        # Decoder usually has two inputs: decoder_input_ids and encoder_hidden_states
        din = self.dec.get_inputs()
        self.dec_in_ids = next(i.name for i in din if "input_ids" in i.name or "decoder_input" in i.name)
        self.dec_in_enc = next(i.name for i in din if "encoder" in i.name)
        self.dec_out_logits = self.dec.get_outputs()[0].name # (1, t, vocab)

        # Offline tokenizer & config
        self.tokenizer = Tokenizer.from_file(str(tok_path))
        cfg = json.load(open(cfg_path, "r", encoding="utf-8"))
        self.decoder_start = int(cfg.get("decoder_start_token_id", 50258))  # <|startoftranscript|>
        self.eos = int(cfg.get("eos_token_id", 50257))
        self.max_len = int(cfg.get("max_length", 224))  # safe cap
        # Optional: notimestamps/language tokens can be added here if you’d like.

    def _encode(self, mel: np.ndarray) -> np.ndarray:
        # mel: [1, 80, 3000]
        out = self.enc.run(None, {self.enc_in: mel})
        return out[0]  # [1, T, D]

    def _decode_greedy(self, enc_states: np.ndarray) -> str:
        # Start with BOS / start-of-transcription token
        ids = [self.decoder_start]
        for _ in range(self.max_len):
            feeds = {
                self.dec_in_ids: np.array(ids, dtype=np.int64)[None, :],
                self.dec_in_enc: enc_states
            }
            logits = self.dec.run(None, feeds)[0]        # [1, t, V]
            next_id = int(logits[0, -1].argmax(-1))
            if next_id == self.eos:
                break
            ids.append(next_id)

        # decode tokens -> text (offline)
        return self.tokenizer.decode(ids[1:], skip_special_tokens=True).strip()

    def transcribe_bytes(self, wav_bytes: bytes) -> str:
        data, sr = sf.read(io.BytesIO(wav_bytes), dtype='float32', always_2d=False)
        if sr != 16000:
            raise ValueError("Whisper expects 16 kHz mono WAV—resample in the UI for best perf.")
        if data.ndim > 1:
            data = data.mean(-1)

        mel = log_mel_spectrogram(data, sr=16000)  # [1, 80, 3000]
        enc = self._encode(mel)
        text = self._decode_greedy(enc)
        return text
