"""
OCR Engine using EasyOCR Detector + Recognizer ONNX with Qualcomm QNN Execution Provider
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
    def __init__(self, 
                detector_path="../models/easyocr_detector/model.onnx",
                recognizer_path="../models/easyocr_recognizer/model.onnx"):
        """
        Initialize OCR engine with separate detector and recognizer models
        
        Args:
            detector_path: Path to EasyOCR detector ONNX model
            recognizer_path: Path to EasyOCR recognizer ONNX model
        """
        self.detector_path = detector_path
        self.recognizer_path = recognizer_path
        self.detector_session = None
        self.recognizer_session = None
        self.npu_available = False
        
        # Initialize ONNX Runtime sessions
        self._initialize_sessions()
    
    def _initialize_sessions(self):
        """Initialize ONNX Runtime sessions for both detector and recognizer"""
        try:
            # Check if model files exist
            if not os.path.exists(self.detector_path):
                raise FileNotFoundError(f"Detector model not found: {self.detector_path}")
            
            if not os.path.exists(self.recognizer_path):
                raise FileNotFoundError(f"Recognizer model not found: {self.recognizer_path}")
            
            # Configure providers - QNN first, CPU as fallback
            providers = ["QNNExecutionProvider"]
            
            # Initialize Detector Session
            try:
                self.detector_session = ort.InferenceSession(
                    self.detector_path,
                    providers=providers
                )
                detector_npu = True
                logger.info("EasyOCR Detector initialized with QNN Execution Provider (NPU)")
                
            except Exception as qnn_error:
                logger.warning(f"QNN not available for detector: {qnn_error}")
                logger.info("Falling back to CPU Execution Provider for detector")
                
                providers_cpu = ["CPUExecutionProvider"]
                self.detector_session = ort.InferenceSession(
                    self.detector_path,
                    providers=providers_cpu
                )
                detector_npu = False
                logger.info("EasyOCR Detector initialized with CPU Execution Provider")
            
            # Initialize Recognizer Session
            try:
                self.recognizer_session = ort.InferenceSession(
                    self.recognizer_path,
                    providers=providers
                )
                recognizer_npu = True
                logger.info("EasyOCR Recognizer initialized with QNN Execution Provider (NPU)")
                
            except Exception as qnn_error:
                logger.warning(f"QNN not available for recognizer: {qnn_error}")
                logger.info("Falling back to CPU Execution Provider for recognizer")
                
                providers_cpu = ["CPUExecutionProvider"]
                self.recognizer_session = ort.InferenceSession(
                    self.recognizer_path,
                    providers=providers_cpu
                )
                recognizer_npu = False
                logger.info("EasyOCR Recognizer initialized with CPU Execution Provider")
            
            # NPU available if both models use NPU
            self.npu_available = detector_npu and recognizer_npu
            
            # Get input/output names
            self.detector_input_name = self.detector_session.get_inputs()[0].name
            self.detector_output_names = [output.name for output in self.detector_session.get_outputs()]
            
            self.recognizer_input_name = self.recognizer_session.get_inputs()[0].name
            self.recognizer_output_names = [output.name for output in self.recognizer_session.get_outputs()]
            
            logger.info(f"Detector - Input: {self.detector_input_name}, Outputs: {self.detector_output_names}")
            logger.info(f"Recognizer - Input: {self.recognizer_input_name}, Outputs: {self.recognizer_output_names}")
            
        except Exception as e:
            logger.error(f"Failed to initialize OCR engine: {e}")
            raise
    
    def preprocess_image(self, image):
        """
        Preprocess image for OCR detection
        
        Args:
            image: OpenCV image (BGR format)
            
        Returns:
            Preprocessed image tensor
        """
        # Convert BGR to RGB
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
        
        # Resize to model's expected input size: 608x800
        target_size = (800, 608)  # (width, height)
        image_resized = cv2.resize(image_rgb, target_size)
        
        # Normalize to [0, 1] range
        image_normalized = image_resized.astype(np.float32) / 255.0
        
        # Convert from HWC to CHW format and add batch dimension [1, C, H, W]
        image_tensor = np.transpose(image_normalized, (2, 0, 1))  # HWC -> CHW
        image_tensor = np.expand_dims(image_tensor, axis=0)       # Add batch dim
        
        return image_tensor
    
    def run_detector(self, image_tensor):
        """
        Run the detector model to find text regions
        
        Args:
            image_tensor: Preprocessed image tensor
            
        Returns:
            Detected text regions (bounding boxes)
        """
        try:
            # Run detector inference
            detector_outputs = self.detector_session.run(
                self.detector_output_names,
                {self.detector_input_name: image_tensor}
            )
            
            # For now, return dummy bounding boxes
            # In a real implementation, you would parse the detector outputs
            # to extract actual bounding box coordinates
            dummy_boxes = [
                [50, 50, 200, 100],   # [x1, y1, x2, y2] format
                [50, 120, 300, 170],
                [50, 190, 250, 240]
            ]
            
            return dummy_boxes
            
        except Exception as e:
            logger.error(f"Detector inference failed: {e}")
            raise
    
    def run_recognizer(self, cropped_regions):
        """
        Run the recognizer model on cropped text regions
        
        Args:
            cropped_regions: List of cropped image regions containing text
            
        Returns:
            Recognized text for each region
        """
        try:
            recognized_texts = []
            
            for region in cropped_regions:
                # Preprocess the cropped region for recognition
                # This would typically involve resizing to recognizer input size
                region_tensor = self.preprocess_region_for_recognition(region)
                
                # Run recognizer inference
                recognizer_outputs = self.recognizer_session.run(
                    self.recognizer_output_names,
                    {self.recognizer_input_name: region_tensor}
                )
                
                # For now, return dummy text
                # In a real implementation, you would decode the recognizer outputs
                # to extract actual text using the model's vocabulary
                dummy_text = f"Sample text region {len(recognized_texts) + 1}"
                recognized_texts.append(dummy_text)
            
            return recognized_texts
            
        except Exception as e:
            logger.error(f"Recognizer inference failed: {e}")
            raise
    
    def preprocess_region_for_recognition(self, region):
        """
        Preprocess cropped region for text recognition
        
        Args:
            region: Cropped image region
            
        Returns:
            Preprocessed tensor for recognition
        """
        # Resize to recognizer input size (typically rectangular, e.g., 100x32)
        target_size = (1000, 64)  # Width x Height for text recognition (model expects 64x1000)  # Width x Height for text recognition
        region_resized = cv2.resize(region, target_size)
        
        # Convert to grayscale if needed
        if len(region_resized.shape) == 3:
            region_gray = cv2.cvtColor(region_resized, cv2.COLOR_BGR2GRAY)
        else:
            region_gray = region_resized
        
        # Normalize
        region_normalized = region_gray.astype(np.float32) / 255.0
        
        # Add batch and channel dimensions [1, 1, H, W]
        region_tensor = np.expand_dims(np.expand_dims(region_normalized, axis=0), axis=0)
        
        return region_tensor
    
    def extract_text(self, image):
        """
        Extract text from image using detector + recognizer pipeline
        
        Args:
            image: OpenCV image (BGR format)
            
        Returns:
            Dictionary with extracted text, confidence, and metadata
        """
        try:
            start_time = time.time()
            
            # Step 1: Preprocess image for detection
            image_tensor = self.preprocess_image(image)
            
            # Step 2: Run detector to find text regions
            text_boxes = self.run_detector(image_tensor)
            
            # Step 3: Crop regions from original image
            cropped_regions = []
            for box in text_boxes:
                x1, y1, x2, y2 = box
                # Ensure coordinates are within image bounds
                h, w = image.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                cropped_region = image[y1:y2, x1:x2]
                if cropped_region.size > 0:  # Only add non-empty regions
                    cropped_regions.append(cropped_region)
            
            # Step 4: Run recognizer on cropped regions
            if cropped_regions:
                recognized_texts = self.run_recognizer(cropped_regions)
                combined_text = " ".join(recognized_texts)
                confidence = 0.90  # Dummy confidence
            else:
                combined_text = "No text detected"
                confidence = 0.0
            
            inference_time = (time.time() - start_time) * 1000  # Convert to ms
            
            return {
                'text': combined_text,
                'confidence': confidence,
                'npu_used': self.npu_available,
                'inference_time': inference_time,
                'regions_found': len(text_boxes),
                'model_info': {
                    'detector': os.path.basename(self.detector_path),
                    'recognizer': os.path.basename(self.recognizer_path),
                    'architecture': 'detector-recognizer'
                }
            }
            
        except Exception as e:
            logger.error(f"Text extraction failed: {e}")
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
            'architecture': 'EasyOCR Detector-Recognizer'
        }