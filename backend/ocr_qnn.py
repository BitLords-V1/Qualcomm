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
    def __init__(self, detector_model_path: str, recognizer_model_path: str):
        """
        Initialize OCR engine with QNN Execution Provider
        
        Args:
            model_path: Path to EasyOCR ONNX model compiled for QNN
        """
        self.detector_model_path = detector_model_path
        self.recognizer_model_path = recognizer_model_path
        self.detector_session = None
        self.recognizer_session = None
        self.npu_available = False
        self.detector_input_name = None
        self.detector_output_names = None
        self.recognizer_input_name = None
        self.recognizer_output_names = None
        
        # Initialize ONNX Runtime session
        self._initialize_session()
    
    def _initialize_session(self):
        """Initialize ONNX Runtime session with QNN Execution Provider"""
        try:
            if not os.path.exists(self.detector_model_path) or not os.path.exists(self.recognizer_model_path):
                raise FileNotFoundError(f"Model not found: {self.detector_model_path} or {self.recognizer_model_path}")
            
            # Configure providers - QNN first, CPU as fallback
            providers = ["QNNExecutionProvider"]
            
            # Add CPU fallback only if QNN is not available
            try:
                self.detector_session = ort.InferenceSession(
                    self.detector_model_path,
                    providers=providers
                )
                self.npu_available = True
                logger.info("OCR detector engine initialized with QNN Execution Provider (NPU)")
                
            except Exception as qnn_error:
                logger.warning(f"QNN not available for detector: {qnn_error}")
                logger.info("Falling back to CPU Execution Provider for detector")
                
                providers = ["CPUExecutionProvider"]
                self.detector_session = ort.InferenceSession(
                    self.detector_model_path,
                    providers=providers
                )
                self.npu_available = False
                logger.info("OCR detector engine initialized with CPU Execution Provider")
            
            try:
                self.recognizer_session = ort.InferenceSession(
                    self.recognizer_model_path,
                    providers=providers
                )
                # Assuming recognizer also uses QNN if detector does
                if self.npu_available: 
                    logger.info("OCR recognizer engine initialized with QNN Execution Provider (NPU)")
                else:
                    logger.info("OCR recognizer engine initialized with CPU Execution Provider")
                
            except Exception as qnn_error:
                logger.warning(f"QNN not available for recognizer: {qnn_error}")
                logger.info("Falling back to CPU Execution Provider for recognizer")
                
                providers = ["CPUExecutionProvider"]
                self.recognizer_session = ort.InferenceSession(
                    self.recognizer_model_path,
                    providers=providers
                )
                self.npu_available = False
                logger.info("OCR recognizer engine initialized with CPU Execution Provider")
            
            # Get input/output names
            self.detector_input_name = self.detector_session.get_inputs()[0].name
            self.detector_output_names = [output.name for output in self.detector_session.get_outputs()]
            
            self.recognizer_input_name = self.recognizer_session.get_inputs()[0].name
            self.recognizer_output_names = [output.name for output in self.recognizer_session.get_outputs()]
            
            logger.info(f"OCR models loaded successfully. Detector Input: {self.detector_input_name}, Detector Outputs: {self.detector_output_names}, Recognizer Input: {self.recognizer_input_name}, Recognizer Outputs: {self.recognizer_output_names}")
            
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
    
    def _get_text_regions(self, detector_outputs, original_image):
        """
        Placeholder: Process detector outputs to get text regions (bounding boxes).
        This would involve non-max suppression and mapping coordinates back to the original image.
        """
        logger.warning("_get_text_regions is a placeholder and needs proper implementation.")
        # Dummy implementation: return a single region covering the whole image
        return [original_image] 

    def _preprocess_recognizer_input(self, region_image):
        """
        Placeholder: Preprocess a single text region image for the recognizer model.
        This typically involves resizing to a fixed height, padding, and normalization.
        """
        logger.warning("_preprocess_recognizer_input is a placeholder and needs proper implementation.")
        # Dummy implementation: Resize to a fixed size and normalize
        target_height = 32 # Common height for OCR recognizers
        aspect_ratio = region_image.shape[1] / region_image.shape[0]
        target_width = int(target_height * aspect_ratio)
        resized = cv2.resize(region_image, (target_width, target_height))
        normalized = resized.astype(np.float32) / 255.0
        
        # Add batch and channel dimensions if necessary (e.g., [1, C, H, W] or [1, H, W, C])
        # Assuming [1, H, W, C] for simplicity, adjust based on actual model input
        input_array = np.expand_dims(normalized, axis=0)
        return input_array

    def _postprocess_recognizer_output(self, recognizer_outputs):
        """
        Placeholder: Postprocess recognizer outputs to extract text and confidence.
        This would involve decoding character probabilities to text.
        """
        logger.warning("_postprocess_recognizer_output is a placeholder and needs proper implementation.")
        # Dummy implementation
        return "Detected Text", 0.90
    
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
            
            # Run detector inference
            detector_outputs = self.detector_session.run(
                self.detector_output_names,
                {self.detector_input_name: input_array}
            )
            
            # Placeholder for processing detector outputs to get text regions
            # In a real implementation, this would involve NMS and bounding box parsing
            text_regions = self._get_text_regions(detector_outputs, image)
            
            recognized_texts = []
            for region_image in text_regions:
                # Preprocess region for recognizer
                recognizer_input = self._preprocess_recognizer_input(region_image)
                
                # Run recognizer inference
                recognizer_outputs = self.recognizer_session.run(
                    self.recognizer_output_names,
                    {self.recognizer_input_name: recognizer_input}
                )
                
                # Postprocess recognizer outputs
                recognized_text, text_confidence = self._postprocess_recognizer_output(recognizer_outputs)
                recognized_texts.append({'text': recognized_text, 'confidence': text_confidence})
            
            # Combine results
            combined_text = " ".join([t['text'] for t in recognized_texts]) if recognized_texts else ""
            combined_confidence = np.mean([t['confidence'] for t in recognized_texts]) if recognized_texts else 0.0
            
            result = {'text': combined_text, 'confidence': combined_confidence}
            
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
            'model_path': self.detector_model_path,
            'providers': self.session.get_providers(),
            'npu_available': self.npu_available,
            'input_shape': self.session.get_inputs()[0].shape,
            'output_shapes': [output.shape for output in self.session.get_outputs()]
        }
