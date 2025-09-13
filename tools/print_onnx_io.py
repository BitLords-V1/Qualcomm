# tools/print_onnx_io.py
import onnxruntime as ort, glob
for p in [
    "models/easyocr_detector.onnx",
    "models/easyocr_recognizer.onnx",
    "models/whisper/encoder.onnx",
    "models/whisper/decoder.onnx",
    "models/trocr_encoder.onnx",
    "models/trocr_decoder.onnx",
]:
    try:
        s = ort.InferenceSession(p, providers=["QNNExecutionProvider","CPUExecutionProvider"])
        print("\n==", p, "==")
        print("Inputs :", [(i.name, i.shape, i.type) for i in s.get_inputs()])
        print("Outputs:", [(o.name, o.shape, o.type) for o in s.get_outputs()])
    except Exception as e:
        print("\n==", p, "== ERROR:", e)
