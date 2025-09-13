"""
Whisper Small Engine using HfWhisperEncoder + HfWhisperDecoder ONNX with Qualcomm QNN Execution Provider
Optimized for NPU acceleration on Windows-on-Snapdragon
"""

import os
import time
import logging
import numpy as np
import wave
import io
import onnxruntime as ort

logger = logging.getLogger(__name__)

class WhisperSmallEngine:
    def __init__(self, 
             encoder_path="../models/model.onnx 2/model.onnx", 
             decoder_path="../models/model.onnx 3/model.onnx"):
        """
        Initialize Whisper Small engine with separate encoder and decoder models
        
        Args:
            encoder_path: Path to HfWhisperEncoder ONNX model
            decoder_path: Path to HfWhisperDecoder ONNX model
        """
        self.encoder_path = encoder_path
        self.decoder_path = decoder_path
        self.encoder_session = None
        self.decoder_session = None
        self.npu_available = False
        
        # Whisper Small model parameters (based on Qualcomm AI Hub specs)
        self.sample_rate = 16000
        self.n_mels = 80
        self.n_ctx = 448
        self.n_vocab = 51865
        self.max_tokens = 200  # Max decoded sequence length
        
        # Model dimensions for Whisper Small
        self.n_audio = 512    # Audio encoder dimension (Small: 512, Tiny: 384)
        self.n_text = 768     # Text decoder dimension (Small: 768, Tiny: 384)
        
        # Initialize ONNX Runtime sessions
        self._initialize_sessions()
    
    def _initialize_sessions(self):
        """Initialize ONNX Runtime sessions for both encoder and decoder"""
        try:
            # Initialize Encoder
            if not os.path.exists(self.encoder_path):
                raise FileNotFoundError(f"Encoder model not found: {self.encoder_path}")
            
            if not os.path.exists(self.decoder_path):
                raise FileNotFoundError(f"Decoder model not found: {self.decoder_path}")
            
            # Configure providers - QNN first, CPU as fallback
            providers = ["QNNExecutionProvider"]
            
            # Initialize Encoder Session
            try:
                self.encoder_session = ort.InferenceSession(
                    self.encoder_path,
                    providers=providers
                )
                encoder_npu = True
                logger.info("Whisper Small Encoder initialized with QNN Execution Provider (NPU)")
                
            except Exception as qnn_error:
                logger.warning(f"QNN not available for encoder: {qnn_error}")
                logger.info("Falling back to CPU Execution Provider for encoder")
                
                try:
                    providers_cpu = ["CPUExecutionProvider"]
                    self.encoder_session = ort.InferenceSession(
                        self.encoder_path,
                        providers=providers_cpu
                    )
                    encoder_npu = False
                    logger.info("Whisper Small Encoder initialized with CPU Execution Provider")
                except Exception as cpu_error:
                    logger.error(f"CPU fallback failed for encoder: {cpu_error}")
                    logger.error("Encoder model appears to be QNN-specific and incompatible with CPU")
                    raise
            
            # Initialize Decoder Session
            try:
                self.decoder_session = ort.InferenceSession(
                    self.decoder_path,
                    providers=providers
                )
                decoder_npu = True
                logger.info("Whisper Small Decoder initialized with QNN Execution Provider (NPU)")
                
            except Exception as qnn_error:
                logger.warning(f"QNN not available for decoder: {qnn_error}")
                logger.info("Falling back to CPU Execution Provider for decoder")
                
                try:
                    providers_cpu = ["CPUExecutionProvider"]
                    self.decoder_session = ort.InferenceSession(
                        self.decoder_path,
                        providers=providers_cpu
                    )
                    decoder_npu = False
                    logger.info("Whisper Small Decoder initialized with CPU Execution Provider")
                except Exception as cpu_error:
                    logger.error(f"CPU fallback failed for decoder: {cpu_error}")
                    logger.error("Decoder model appears to be QNN-specific and incompatible with CPU")
                    raise
            
            # NPU available if both models use NPU
            self.npu_available = encoder_npu and decoder_npu
            
            # Get input/output names
            self.encoder_input_name = self.encoder_session.get_inputs()[0].name
            self.encoder_output_names = [output.name for output in self.encoder_session.get_outputs()]
            
            self.decoder_input_names = [input.name for input in self.decoder_session.get_inputs()]
            self.decoder_output_names = [output.name for output in self.decoder_session.get_outputs()]
            
            logger.info(f"Encoder - Input: {self.encoder_input_name}, Outputs: {self.encoder_output_names}")
            logger.info(f"Decoder - Inputs: {self.decoder_input_names}, Outputs: {self.decoder_output_names}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Whisper Small engine: {e}")
            raise
    
    def transcribe(self, audio_bytes):
        """
        Transcribe audio using Whisper Small encoder-decoder models
        
        Args:
            audio_bytes: Raw audio bytes (WAV format)
            
        Returns:
            Dictionary with transcribed text, confidence, and metadata
        """
        try:
            start_time = time.time()
            
            # For now, return a test response to verify integration
            inference_time = (time.time() - start_time) * 1000
            
            return {
                'text': f"Whisper Small test transcription (NPU: {self.npu_available})",
                'confidence': 0.95,
                'npu_used': self.npu_available,
                'inference_time': inference_time,
                'model_info': {
                    'encoder': os.path.basename(self.encoder_path),
                    'decoder': os.path.basename(self.decoder_path),
                    'architecture': 'encoder-decoder'
                }
            }
            
        except Exception as e:
            logger.error(f"Speech transcription failed: {e}")
            return {
                'text': '',
                'confidence': 0.0,
                'npu_used': self.npu_available,
                'inference_time': 0.0,
                'error': str(e)
            }
    
    def get_model_info(self):
        """Get information about the loaded models"""
        if not self.encoder_session or not self.decoder_session:
            return None
            
        return {
            'encoder_path': self.encoder_path,
            'decoder_path': self.decoder_path,
            'npu_available': self.npu_available,
            'architecture': 'Whisper Small Encoder-Decoder'
        }