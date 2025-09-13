import onnxruntime as ort

def make_qnn_session(model_path: str, strict_qnn: bool = False) -> ort.InferenceSession:
    """
    Create an ONNX Runtime session that prefers the Qualcomm NPU (QNN EP) and
    falls back to CPU if needed. Set strict_qnn=True during bring-up to ensure
    nothing falls back silently.
    """
    # Good for bring-up: force failure if a node would fall back to CPU
    so = ort.SessionOptions()
    if strict_qnn:
        so.add_session_config_entry("session.disable_cpu_ep_fallback", "1")

    return ort.InferenceSession(
        model_path,
        sess_options=so,
        providers=["QNNExecutionProvider", "CPUExecutionProvider"],
        provider_options=[{
            "backend_path": "QnnHtp.dll",
            "htp_performance_mode": "sustained_high_performance"
        }, {}]
    )
