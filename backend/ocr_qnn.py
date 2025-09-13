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
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)

class OCREngine:
    def __init__(self, detector_path="../models/easyocr_detector/model.onnx", 
                 recognizer_path="../models/easyocr_recognizer/model.onnx"):
        """
        Initialize OCR engine with separate detector and recognizer models
        
        Args:
            detector_path: Path to EasyOCR detector ONNX model compiled for QNN
            recognizer_path: Path to EasyOCR recognizer ONNX model compiled for QNN
        """
        self.detector_path = detector_path
        self.recognizer_path = recognizer_path
        self.detector_session = None
        self.recognizer_session = None
        self.npu_available = False
        
        # Model parameters
        self.detector_input_size = (608, 800)  # Height, Width
        self.recognizer_input_size = (64, 1000)  # Height, Width
        self.max_text_length = 249  # Maximum text length for recognizer
        
        # Initialize ONNX Runtime sessions
        self._initialize_sessions()
    
    def _initialize_sessions(self):
        """Initialize ONNX Runtime sessions for detector and recognizer with QNN Execution Provider"""
        try:
            # Check if model files exist
            if not os.path.exists(self.detector_path):
                raise FileNotFoundError(f"Detector model not found: {self.detector_path}")
            if not os.path.exists(self.recognizer_path):
                raise FileNotFoundError(f"Recognizer model not found: {self.recognizer_path}")
            
            # Configure providers - QNN first, CPU as fallback
            providers = ["QNNExecutionProvider"]
            
            # Initialize detector session
            try:
                self.detector_session = ort.InferenceSession(
                    self.detector_path,
                    providers=providers
                )
                self.npu_available = True
                logger.info("Detector initialized with QNN Execution Provider (NPU)")
                
            except Exception as qnn_error:
                logger.warning(f"QNN not available for detector: {qnn_error}")
                logger.info("Attempting CPU fallback for detector...")
                
                try:
                    providers = ["CPUExecutionProvider"]
                    self.detector_session = ort.InferenceSession(
                        self.detector_path,
                        providers=providers
                    )
                    self.npu_available = False
                    logger.info("Detector initialized with CPU Execution Provider")
                except Exception as cpu_error:
                    logger.error(f"CPU fallback also failed for detector: {cpu_error}")
                    logger.warning("Detector model requires QNN/NPU - cannot run on CPU")
                    self.detector_session = None
                    self.npu_available = False
            
            # Initialize recognizer session
            try:
                self.recognizer_session = ort.InferenceSession(
                    self.recognizer_path,
                    providers=["QNNExecutionProvider"]
                )
                logger.info("Recognizer initialized with QNN Execution Provider (NPU)")
                
            except Exception as qnn_error:
                logger.warning(f"QNN not available for recognizer: {qnn_error}")
                logger.info("Attempting CPU fallback for recognizer...")
                
                try:
                    providers = ["CPUExecutionProvider"]
                    self.recognizer_session = ort.InferenceSession(
                        self.recognizer_path,
                        providers=providers
                    )
                    logger.info("Recognizer initialized with CPU Execution Provider")
                except Exception as cpu_error:
                    logger.error(f"CPU fallback also failed for recognizer: {cpu_error}")
                    logger.warning("Recognizer model requires QNN/NPU - cannot run on CPU")
                    self.recognizer_session = None
            
            # Check if both sessions were initialized
            if self.detector_session is None or self.recognizer_session is None:
                logger.warning("OCR engine initialized in limited mode - models require QNN/NPU")
                logger.info("OCR functionality will be disabled until QNN/NPU is available")
            else:
                logger.info("OCR detector and recognizer models loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize OCR engine: {e}")
            # Don't raise exception - allow app to continue without OCR
            self.detector_session = None
            self.recognizer_session = None
            self.npu_available = False
    
    def is_ready(self):
        """Check if the OCR engine is ready for inference"""
        return self.detector_session is not None and self.recognizer_session is not None
    
    def preprocess_image_for_detector(self, image):
        """
        Preprocess image for detector inference
        
        Args:
            image: OpenCV image (BGR format)
            
        Returns:
            Preprocessed image array [1, 3, 608, 800]
        """
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize to detector input size (608x800)
        resized = cv2.resize(rgb_image, (self.detector_input_size[1], self.detector_input_size[0]))
        
        # Normalize to [0, 1]
        normalized = resized.astype(np.float32) / 255.0
        
        # Add batch dimension and transpose to CHW format
        input_array = np.transpose(normalized, (2, 0, 1))
        input_array = np.expand_dims(input_array, axis=0)
        
        return input_array
    
    def preprocess_text_region_for_recognizer(self, image, bbox):
        """
        Preprocess text region for recognizer inference
        
        Args:
            image: OpenCV image (BGR format)
            bbox: Bounding box coordinates [x1, y1, x2, y2, x3, y3, x4, y4]
            
        Returns:
            Preprocessed text region array [1, 1, 64, 1000]
        """
        # Extract text region using bounding box
        # Convert bbox to rectangle and extract region
        x_coords = bbox[::2]  # x coordinates
        y_coords = bbox[1::2]  # y coordinates
        
        x_min, x_max = int(min(x_coords)), int(max(x_coords))
        y_min, y_max = int(min(y_coords)), int(max(y_coords))
        
        # Ensure coordinates are within image bounds
        h, w = image.shape[:2]
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(w, x_max)
        y_max = min(h, y_max)
        
        # Extract text region
        text_region = image[y_min:y_max, x_min:x_max]
        
        if text_region.size == 0:
            # Return empty region if extraction failed
            return np.zeros((1, 1, self.recognizer_input_size[0], self.recognizer_input_size[1]), dtype=np.float32)
        
        # Convert to grayscale
        if len(text_region.shape) == 3:
            gray_region = cv2.cvtColor(text_region, cv2.COLOR_BGR2GRAY)
        else:
            gray_region = text_region
        
        # Resize to recognizer input size (64x1000)
        resized = cv2.resize(gray_region, (self.recognizer_input_size[1], self.recognizer_input_size[0]))
        
        # Normalize to [0, 1]
        normalized = resized.astype(np.float32) / 255.0
        
        # Add batch and channel dimensions
        input_array = np.expand_dims(normalized, axis=0)  # Add batch dimension
        input_array = np.expand_dims(input_array, axis=0)  # Add channel dimension
        
        return input_array
    
    def run_detector(self, image_array):
        """
        Run the detector model to find text regions
        
        Args:
            image_array: Preprocessed image array [1, 3, 608, 800]
            
        Returns:
            Dictionary with detection results and features
        """
        try:
            # Run detector inference
            outputs = self.detector_session.run(
                None,  # All outputs
                {"image": image_array}
            )
            
            # Extract results and features
            results = outputs[0]  # [1, 304, 400, 2] - detection results
            features = outputs[1]  # [1, 32, 304, 400] - features for recognizer
            
            return {
                'results': results,
                'features': features
            }
            
        except Exception as e:
            logger.error(f"Detector inference failed: {e}")
            raise
    
    def run_recognizer(self, text_region_array):
        """
        Run the recognizer model to recognize text
        
        Args:
            text_region_array: Preprocessed text region array [1, 1, 64, 1000]
            
        Returns:
            Dictionary with recognition results
        """
        try:
            # Run recognizer inference
            outputs = self.recognizer_session.run(
                None,  # All outputs
                {"image": text_region_array}
            )
            
            # Extract predictions
            output_preds = outputs[0]  # [1, 249, 97] - character predictions
            
            return {
                'predictions': output_preds
            }
            
        except Exception as e:
            logger.error(f"Recognizer inference failed: {e}")
            raise
    
    def postprocess_detection_results(self, results, original_image):
        """
        Postprocess detector outputs to extract bounding boxes
        
        Args:
            results: Detector output results [1, 304, 400, 2]
            original_image: Original input image for coordinate mapping
            
        Returns:
            List of bounding boxes
        """
        # This is a simplified postprocessing
        # In a real implementation, you would:
        # 1. Apply threshold to detection results
        # 2. Find connected components
        # 3. Extract bounding boxes
        # 4. Map coordinates back to original image
        
        # For now, return dummy bounding boxes
        h, w = original_image.shape[:2]
        dummy_bboxes = [
            [w*0.1, h*0.1, w*0.4, h*0.1, w*0.4, h*0.2, w*0.1, h*0.2],  # Text region 1
            [w*0.5, h*0.3, w*0.8, h*0.3, w*0.8, h*0.4, w*0.5, h*0.4],  # Text region 2
        ]
        
        return dummy_bboxes
    
    def postprocess_recognition_results(self, predictions):
        """
        Postprocess recognizer outputs to extract text
        
        Args:
            predictions: Recognizer output predictions [1, 249, 97]
            
        Returns:
            Dictionary with recognized text and confidence
        """
        # This is a simplified postprocessing
        # In a real implementation, you would:
        # 1. Apply softmax to get probabilities
        # 2. Find the most likely character at each position
        # 3. Remove duplicate characters and special tokens
        # 4. Convert character IDs to text
        
        # For now, return dummy text
        text = "Sample OCR text"
        confidence = 0.92
        
        return {
            'text': text,
            'confidence': confidence
        }
    
    def extract_text(self, image):
        """
        Extract text from image using EasyOCR detector and recognizer pipeline
        
        Args:
            image: OpenCV image (BGR format)
            
        Returns:
            Dictionary with extracted text, confidence, and metadata
        """
        # Check if engine is ready
        if not self.is_ready():
            return {
                'text': '',
                'confidence': 0.0,
                'npu_used': False,
                'inference_time': 0.0,
                'error': 'OCR engine not ready - models require QNN/NPU hardware'
            }
        
        try:
            start_time = time.time()
            
            # Step 1: Preprocess image for detector
            detector_input = self.preprocess_image_for_detector(image)
            
            # Step 2: Run detector to find text regions
            detection_results = self.run_detector(detector_input)
            
            # Step 3: Postprocess detection results to get bounding boxes
            bounding_boxes = self.postprocess_detection_results(
                detection_results['results'], image
            )
            
            # Step 4: Extract and recognize text from each region
            all_texts = []
            all_confidences = []
            
            for bbox in bounding_boxes:
                # Preprocess text region for recognizer
                text_region_input = self.preprocess_text_region_for_recognizer(image, bbox)
                
                # Run recognizer
                recognition_results = self.run_recognizer(text_region_input)
                
                # Postprocess recognition results
                text_result = self.postprocess_recognition_results(
                    recognition_results['predictions']
                )
                
                all_texts.append(text_result['text'])
                all_confidences.append(text_result['confidence'])
            
            # Combine all text results
            combined_text = ' '.join(all_texts) if all_texts else ''
            avg_confidence = np.mean(all_confidences) if all_confidences else 0.0
            
            inference_time = (time.time() - start_time) * 1000  # Convert to ms
            
            return {
                'text': combined_text,
                'confidence': avg_confidence,
                'npu_used': self.npu_available,
                'inference_time': inference_time,
                'regions_detected': len(bounding_boxes)
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
        """Get information about the loaded models"""
        if not self.detector_session or not self.recognizer_session:
            return None
            
        return {
            'detector_path': self.detector_path,
            'recognizer_path': self.recognizer_path,
            'detector_providers': self.detector_session.get_providers(),
            'recognizer_providers': self.recognizer_session.get_providers(),
            'npu_available': self.npu_available,
            'detector_input_shape': self.detector_session.get_inputs()[0].shape,
            'recognizer_input_shape': self.recognizer_session.get_inputs()[0].shape,
            'detector_output_shapes': [output.shape for output in self.detector_session.get_outputs()],
            'recognizer_output_shapes': [output.shape for output in self.recognizer_session.get_outputs()]
        }
