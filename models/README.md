# iLumina Models Directory

This directory contains the ONNX models required for iLumina to function.

## Required Models

### 1. EasyOCR ONNX Model
- **File**: `easyocr_qnn.onnx`
- **Source**: [Qualcomm AI Hub](https://aihub.qualcomm.com)
- **Purpose**: Text extraction from images using OCR
- **Target**: Qualcomm NPU (QNN Execution Provider)

### 2. Whisper Tiny EN ONNX Model
- **File**: `whisper_tiny_en_qnn.onnx`
- **Source**: [Qualcomm AI Hub](https://aihub.qualcomm.com)
- **Purpose**: Speech-to-text conversion
- **Target**: Qualcomm NPU (QNN Execution Provider)

## Model Setup Instructions

1. **Download Models**:
   - Visit [Qualcomm AI Hub](https://aihub.qualcomm.com)
   - Search for "EasyOCR" and download the QNN-compiled version
   - Search for "Whisper Tiny EN" and download the QNN-compiled version
   - Ensure models are compiled for Windows-on-Snapdragon (ARM64)

2. **Place Models**:
   - Copy `easyocr_qnn.onnx` to this directory
   - Copy `whisper_tiny_en_qnn.onnx` to this directory

3. **Verify Models**:
   - Models should be optimized for QNN Execution Provider
   - File sizes should be reasonable (typically 10-100MB each)
   - Models should be compatible with ONNX Runtime 1.16.3+

## Model Requirements

- **Format**: ONNX (Open Neural Network Exchange)
- **Target**: Qualcomm QNN Execution Provider
- **Architecture**: ARM64 (Windows-on-Snapdragon)
- **Precision**: FP16 or FP32 (QNN optimized)

## Troubleshooting

### Model Not Found
- Ensure models are placed in the correct directory
- Check file names match exactly (case-sensitive)
- Verify file permissions allow reading

### QNN Not Available
- Install Qualcomm QNN drivers
- Ensure Windows-on-Snapdragon device
- Check ONNX Runtime QNN provider installation

### Model Loading Errors
- Verify model compatibility with ONNX Runtime version
- Check model file integrity
- Ensure models are QNN-compiled, not CPU-only

## Performance Notes

- **NPU Acceleration**: Models will automatically use Qualcomm NPU when available
- **CPU Fallback**: If NPU is not available, models will fall back to CPU execution
- **Inference Time**: NPU typically provides 2-5x speedup over CPU
- **Memory Usage**: NPU models may use more memory but provide better performance

## Support

For model-related issues:
1. Check Qualcomm AI Hub documentation
2. Verify QNN driver installation
3. Test with sample models from Qualcomm
4. Contact Qualcomm support for model-specific issues
