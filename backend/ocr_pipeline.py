"""
OCR Engine using EasyOCR ONNX with Qualcomm QNN Execution Provider
Optimized for NPU acceleration on Windows-on-Snapdragon
"""

import os
import time
import logging
import numpy as np
import cv2
from PIL import Image
import onnxruntime as ort

logger = logging.getLogger(__name__)

class OCREngine:
    def __init__(self, model_path="models/easyocr_qnn.onnx"):
        """
        Initialize OCR engine with QNN Execution Provider
        
        Args:
            model_path: Path to EasyOCR ONNX model compiled for QNN
        """
        self.model_path = model_path
        self.session = None
        self.npu_available = False
        self.input_name = None
        self.output_names = None
        
        # Initialize ONNX Runtime session
        self._initialize_session()
    
    def _initialize_session(self):
        """Initialize ONNX Runtime session with QNN Execution Provider"""
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model not found: {self.model_path}")
            
            # Configure providers - QNN first, CPU as fallback
            providers = ["QNNExecutionProvider"]
            
            # Add CPU fallback only if QNN is not available
            try:
                self.session = ort.InferenceSession(
                    self.model_path,
                    providers=providers
                )
                self.npu_available = True
                logger.info("OCR engine initialized with QNN Execution Provider (NPU)")
                
            except Exception as qnn_error:
                logger.warning(f"QNN not available: {qnn_error}")
                logger.info("Falling back to CPU Execution Provider")
                
                providers = ["CPUExecutionProvider"]
                self.session = ort.InferenceSession(
                    self.model_path,
                    providers=providers
                )
                self.npu_available = False
                logger.info("OCR engine initialized with CPU Execution Provider")
            
            # Get input/output names
            self.input_name = self.session.get_inputs()[0].name
            self.output_names = [output.name for output in self.session.get_outputs()]
            
            logger.info(f"OCR model loaded successfully. Input: {self.input_name}, Outputs: {self.output_names}")
            
        except Exception as e:
            logger.error(f"Failed to initialize OCR engine: {e}")
            raise
    
    def preprocess_image(self, image):
        """
        Preprocess image for OCR inference
        
        Args:
            image: OpenCV image (BGR format)
            
        Returns:
            Preprocessed image array
        """
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize to model input size (typically 640x640 for EasyOCR)
        target_size = (640, 640)
        resized = cv2.resize(rgb_image, target_size)
        
        # Normalize to [0, 1]
        normalized = resized.astype(np.float32) / 255.0
        
        # Add batch dimension and transpose to CHW format
        input_array = np.transpose(normalized, (2, 0, 1))
        input_array = np.expand_dims(input_array, axis=0)
        
        return input_array
    
    def postprocess_output(self, outputs, original_image):
        """
        Postprocess OCR model outputs to extract text
        
        Args:
            outputs: Model outputs
            original_image: Original input image for coordinate mapping
            
        Returns:
            Dictionary with extracted text and confidence
        """
        # This is a simplified postprocessing
        # In a real implementation, you would parse the model outputs
        # to extract bounding boxes, text, and confidence scores
        
        # For now, return a placeholder implementation
        # The actual implementation would depend on the specific EasyOCR ONNX model structure
        
        text = "Sample OCR text extraction"
        confidence = 0.95
        
        return {
            'text': text,
            'confidence': confidence
        }
    
    def extract_text(self, image):
        """
        Extract text from image using EasyOCR ONNX model
        
        Args:
            image: OpenCV image (BGR format)
            
        Returns:
            Dictionary with extracted text, confidence, and metadata
        """
        try:
            start_time = time.time()
            
            # Preprocess image
            input_array = self.preprocess_image(image)
            
            # Run inference
            outputs = self.session.run(
                self.output_names,
                {self.input_name: input_array}
            )
            
            # Postprocess outputs
            result = self.postprocess_output(outputs, image)
            
            inference_time = (time.time() - start_time) * 1000  # Convert to ms
            
            return {
                'text': result['text'],
                'confidence': result['confidence'],
                'npu_used': self.npu_available,
                'inference_time': inference_time
            }
            
        except Exception as e:
            logger.error(f"OCR extraction failed: {e}")
            return {
                'text': '',
                'confidence': 0.0,
                'npu_used': self.npu_available,
                'inference_time': 0.0,
                'error': str(e)
            }
    
    def get_model_info(self):
        """Get information about the loaded model"""
        if not self.session:
            return None
            
        return {
            'model_path': self.model_path,
            'providers': self.session.get_providers(),
            'npu_available': self.npu_available,
            'input_shape': self.session.get_inputs()[0].shape,
            'output_shapes': [output.shape for output in self.session.get_outputs()]
        }
