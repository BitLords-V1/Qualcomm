#!/usr/bin/env python3
"""
Example usage of the updated Whisper encoder-decoder implementation
"""

import sys
import os

# Add backend to path
sys.path.append('backend')

from whisper_qnn import WhisperEngine

def main():
    """Demonstrate how to use the Whisper engine with encoder and decoder models"""
    
    print("Whisper Encoder-Decoder Usage Example")
    print("=" * 50)
    
    # Initialize the Whisper engine with separate encoder and decoder models
    print("1. Initializing Whisper engine...")
    try:
        engine = WhisperEngine(
            encoder_path="models/whisper_encoder/model.onnx",
            decoder_path="models/whisper_decoder/model.onnx"
        )
        print("   ✓ Engine initialized successfully")
    except Exception as e:
        print(f"   ✗ Failed to initialize: {e}")
        return
    
    # Get model information
    print("\n2. Model Information:")
    model_info = engine.get_model_info()
    if model_info:
        print(f"   - Encoder Path: {model_info['encoder_path']}")
        print(f"   - Decoder Path: {model_info['decoder_path']}")
        print(f"   - NPU Available: {model_info['npu_available']}")
        print(f"   - Sample Rate: {model_info['sample_rate']} Hz")
        print(f"   - Encoder Input Shape: {model_info['encoder_input_shape']}")
        print(f"   - Decoder Input Shapes: {len(model_info['decoder_input_shapes'])} inputs")
        print(f"   - Encoder Output Shapes: {len(model_info['encoder_output_shapes'])} outputs")
        print(f"   - Decoder Output Shapes: {len(model_info['decoder_output_shapes'])} outputs")
    
    print("\n3. How the Speech-to-Text Pipeline Works:")
    print("   a) Audio is loaded and converted to mel spectrogram [1, 80, 3000]")
    print("   b) Encoder processes mel spectrogram and outputs cross-attention caches")
    print("   c) Decoder generates text tokens autoregressively using:")
    print("      - Input token IDs")
    print("      - Attention masks")
    print("      - Cross-attention caches from encoder")
    print("      - Self-attention caches (for autoregressive generation)")
    print("   d) Tokens are decoded to text")
    
    print("\n4. Key Features:")
    print("   - Separate encoder and decoder models for better control")
    print("   - Proper mel spectrogram preprocessing using librosa")
    print("   - Autoregressive text generation with attention caches")
    print("   - QNN/NPU acceleration with CPU fallback")
    print("   - Support for up to 30 seconds of audio")
    
    print("\n5. Usage Example:")
    print("""
    # Load audio file
    with open('audio.wav', 'rb') as f:
        audio_bytes = f.read()
    
    # Transcribe audio
    result = engine.transcribe(audio_bytes)
    
    # Get results
    text = result['text']
    confidence = result['confidence']
    inference_time = result['inference_time']
    npu_used = result['npu_used']
    """)
    
    print("\n6. Model Architecture Details:")
    print("   Encoder:")
    print("   - Input: Mel spectrogram [1, 80, 3000]")
    print("   - Output: Cross-attention key-value caches for 4 layers")
    print("   - Purpose: Extract audio features for text generation")
    print()
    print("   Decoder:")
    print("   - Input: Token IDs, attention masks, cross-attention caches")
    print("   - Output: Logits [1, 51865, 1, 1] and updated self-attention caches")
    print("   - Purpose: Generate text tokens autoregressively")
    
    print("\n7. Important Notes:")
    print("   - Models are compiled for QNN/NPU and require Qualcomm hardware")
    print("   - On CPU, models will fail to load (QNN context not available)")
    print("   - For production use, implement proper Whisper tokenizer")
    print("   - Consider adding beam search for better text quality")
    print("   - Audio should be 16kHz mono WAV format")

if __name__ == "__main__":
    main()
