from __future__ import annotations
from pathlib import Path
from typing import List, Tuple
import numpy as np
from PIL import Image
from .qnn_runtime import make_qnn_session

def _input_hw(sess) -> Tuple[int, int]:
    ih, iw = sess.get_inputs()[0].shape[-2:]
    return (int(ih) if isinstance(ih, int) else 384,
            int(iw) if isinstance(iw, int) else 384)

def _prep_rgb(img: Image.Image, hw: Tuple[int,int]) -> np.ndarray:
    H, W = hw
    x = img.convert("RGB").resize((W, H))
    x = (np.asarray(x, dtype=np.float32) / 255.0).transpose(2,0,1)[None, ...]  # NCHW
    return x

def _maybe_denorm(box: np.ndarray, W: int, H: int) -> np.ndarray:
    """
    If a box looks normalized (0..1), scale to image coords.
    Supports shapes: (4,2), (8,), (4,)
    """
    b = box.astype(np.float32)
    if b.max() <= 1.5:  # likely normalized
        if b.ndim == 1 and b.size == 4:
            x1,y1,x2,y2 = b
            return np.array([[x1*W,y1*H],[x2*W,y1*H],[x2*W,y2*H],[x1*W,y2*H]], dtype=np.float32)
        if b.ndim == 1 and b.size == 8:
            b = b.reshape(4,2); b[:,0]*=W; b[:,1]*=H; return b
        if b.shape == (4,2):
            b[:,0]*=W; b[:,1]*=H; return b
    # already in pixels → transform shapes if needed
    if b.ndim == 1 and b.size == 4:
        x1,y1,x2,y2 = b
        return np.array([[x1,y1],[x2,y1],[x2,y2],[x1,y2]], dtype=np.float32)
    if b.ndim == 1 and b.size == 8:
        return b.reshape(4,2)
    return b  # assume (4,2)

class EasyOCR_QNN:
    """
    Minimal EasyOCR: Detector -> crops -> Recognizer
    Works with Qualcomm AI Hub ONNX exports (device-agnostic).
    """
    def __init__(self,
                 det_path: Path = Path("models/easyocr_detector.onnx"),
                 rec_path: Path = Path("models/easyocr_recognizer.onnx")):
        self.det_sess = make_qnn_session(str(det_path))
        self.rec_sess = make_qnn_session(str(rec_path))
        self.det_in = self.det_sess.get_inputs()[0].name
        self.rec_in = self.rec_sess.get_inputs()[0].name
        self.det_hw = _input_hw(self.det_sess)
        self.rec_hw = _input_hw(self.rec_sess)
        self.det_out_names = [o.name for o in self.det_sess.get_outputs()]
        self.rec_out_names = [o.name for o in self.rec_sess.get_outputs()]

    def _postprocess_detector(self, outs: List[np.ndarray], orig: Image.Image) -> List[np.ndarray]:
        """
        Heuristics to handle common detector outputs from AI Hub exports.
        Target: list of quadrilaterals (4,2) in image coordinates.
        """
        W, H = orig.size
        boxes = outs[0]
        quads: List[np.ndarray] = []

        # Case A: (N,4,2) already quads
        if boxes.ndim == 3 and boxes.shape[-2:] == (4,2):
            return [ _maybe_denorm(b, W, H) for b in boxes ]

        # Case B: (N,4) axis-aligned
        if boxes.ndim == 2 and boxes.shape[1] == 4:
            for b in boxes:
                quads.append(_maybe_denorm(b, W, H))
            return quads

        # Case C: (N,8) flattened quad
        if boxes.ndim == 2 and boxes.shape[1] == 8:
            for b in boxes:
                quads.append(_maybe_denorm(b, W, H))
            return quads

        # Fallback: treat as a single full-image box
        return [np.array([[0,0],[W,0],[W,H],[0,H]], dtype=np.float32)]

    def _decode_recognizer(self, outs: List[np.ndarray]) -> str:
        # Some recognizer exports already return a string tensor:
        if isinstance(outs[0], (str, bytes)):
            return outs[0].decode() if isinstance(outs[0], bytes) else outs[0]

        # Common: logits [N, T, C] -> greedy CTC decode
        logits = outs[0]
        if logits.ndim == 3:
            blank = logits.shape[-1] - 1
            seq = logits[0].argmax(-1)
            text, prev = [], None
            alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-.,:;!?()[]{}'\"/\\@#$%^&*_+=<>|~ "
            for t in seq:
                t = int(t)
                if t != blank and t != prev:
                    text.append(alphabet[t] if t < len(alphabet) else "")
                prev = t
            return "".join(text).strip()

        # Last resort: stringify
        return str(outs[0])

    def infer(self, image: Image.Image) -> str:
        # DETECT
        det_in = _prep_rgb(image, self.det_hw)
        det_out = self.det_sess.run(None, {self.det_in: det_in})
        boxes = self._postprocess_detector(det_out, image)
        if not boxes:
            return ""

        # RECOGNIZE per crop
        lines = []
        for b in sorted(boxes, key=lambda B: (B[:,1].mean(), B[:,0].mean())):
            x0, y0 = b.min(0); x1, y1 = b.max(0)
            x0, y0 = max(0,int(x0)), max(0,int(y0))
            x1, y1 = min(image.width, int(x1)), min(image.height, int(y1))
            if x1 - x0 < 4 or y1 - y0 < 4:  # tiny crop → skip
                continue
            crop = image.crop((x0,y0,x1,y1))
            rec_in = _prep_rgb(crop, self.rec_hw)
            rec_out = self.rec_sess.run(None, {self.rec_in: rec_in})
            text = self._decode_recognizer(rec_out)
            if text:
                lines.append(((y0, x0), text))

        lines.sort(key=lambda p: (p[0][0], p[0][1]))
        return "\n".join(t for _, t in lines)

# Public convenience
from functools import lru_cache

@lru_cache(maxsize=1)
def _easy_engine(): return EasyOCR_QNN()

def ocr_easy(pil_image: Image.Image) -> str:
    return _easy_engine().infer(pil_image)
