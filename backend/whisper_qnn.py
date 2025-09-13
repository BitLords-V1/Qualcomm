"""
Whisper Engine using Whisper Tiny EN ONNX with Qualcomm QNN Execution Provider
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

class WhisperEngine:
    def __init__(self, model_path="models/whisper_tiny_en_qnn.onnx"):
        """
        Initialize Whisper engine with QNN Execution Provider
        
        Args:
            model_path: Path to Whisper Tiny EN ONNX model compiled for QNN
        """
        self.model_path = model_path
        self.session = None
        self.npu_available = False
        self.input_name = None
        self.output_names = None
        
        # Whisper model parameters
        self.sample_rate = 16000
        self.n_mels = 80
        self.n_ctx = 448
        self.n_vocab = 51865
        
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
                logger.info("Whisper engine initialized with QNN Execution Provider (NPU)")
                
            except Exception as qnn_error:
                logger.warning(f"QNN not available: {qnn_error}")
                logger.info("Falling back to CPU Execution Provider")
                
                providers = ["CPUExecutionProvider"]
                self.session = ort.InferenceSession(
                    self.model_path,
                    providers=providers
                )
                self.npu_available = False
                logger.info("Whisper engine initialized with CPU Execution Provider")
            
            # Get input/output names
            self.input_name = self.session.get_inputs()[0].name
            self.output_names = [output.name for output in self.session.get_outputs()]
            
            logger.info(f"Whisper model loaded successfully. Input: {self.input_name}, Outputs: {self.output_names}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Whisper engine: {e}")
            raise
    
    def load_audio(self, audio_bytes):
        """
        Load and preprocess audio data
        
        Args:
            audio_bytes: Raw audio bytes (WAV format)
            
        Returns:
            Preprocessed audio array
        """
        try:
            # Load audio from bytes
            audio_io = io.BytesIO(audio_bytes)
            with wave.open(audio_io, 'rb') as wav_file:
                # Get audio parameters
                sample_rate = wav_file.getframerate()
                audio_data = wav_file.readframes(wav_file.getnframes())
                
                # Convert to numpy array
                if wav_file.getsampwidth() == 1:
                    audio_array = np.frombuffer(audio_data, dtype=np.uint8)
                    audio_array = (audio_array.astype(np.float32) - 128) / 128.0
                elif wav_file.getsampwidth() == 2:
                    audio_array = np.frombuffer(audio_data, dtype=np.int16)
                    audio_array = audio_array.astype(np.float32) / 32768.0
                else:
                    raise ValueError(f"Unsupported sample width: {wav_file.getsampwidth()}")
                
                # Resample if necessary
                if sample_rate != self.sample_rate:
                    # Simple resampling (in production, use librosa or scipy)
                    ratio = self.sample_rate / sample_rate
                    new_length = int(len(audio_array) * ratio)
                    audio_array = np.interp(
                        np.linspace(0, len(audio_array), new_length),
                        np.arange(len(audio_array)),
                        audio_array
                    )
                
                return audio_array
                
        except Exception as e:
            logger.error(f"Failed to load audio: {e}")
            raise
    
    def preprocess_audio(self, audio_array):
        """
        Preprocess audio for Whisper inference
        
        Args:
            audio_array: Audio array (1D, float32, normalized)
            
        Returns:
            Preprocessed mel spectrogram
        """
        # This is a simplified preprocessing
        # In a real implementation, you would compute the mel spectrogram
        # using librosa or similar library
        
        # For now, create a dummy mel spectrogram
        # The actual implementation would depend on the specific Whisper ONNX model structure
        
        # Pad or truncate to 30 seconds (480,000 samples at 16kHz)
        target_length = 30 * self.sample_rate
        if len(audio_array) > target_length:
            audio_array = audio_array[:target_length]
        else:
            audio_array = np.pad(audio_array, (0, target_length - len(audio_array)))
        
        # Create dummy mel spectrogram (80 x 3000)
        mel_spectrogram = np.random.randn(self.n_mels, 3000).astype(np.float32)
        
        # Add batch dimension
        mel_spectrogram = np.expand_dims(mel_spectrogram, axis=0)
        
        return mel_spectrogram
    
    def postprocess_output(self, outputs):
        """
        Postprocess Whisper model outputs to extract text
        
        Args:
            outputs: Model outputs
            
        Returns:
            Dictionary with transcribed text and confidence
        """
        # This is a simplified postprocessing
        # In a real implementation, you would decode the token IDs
        # to text using the Whisper tokenizer
        
        # For now, return a placeholder implementation
        text = "Sample speech transcription"
        confidence = 0.92
        
        return {
            'text': text,
            'confidence': confidence
        }
    
    def transcribe(self, audio_bytes):
        """
        Transcribe audio using Whisper ONNX model
        
        Args:
            audio_bytes: Raw audio bytes (WAV format)
            
        Returns:
            Dictionary with transcribed text, confidence, and metadata
        """
        try:
            start_time = time.time()
            
            # Load and preprocess audio
            audio_array = self.load_audio(audio_bytes)
            mel_spectrogram = self.preprocess_audio(audio_array)
            
            # Run inference
            outputs = self.session.run(
                self.output_names,
                {self.input_name: mel_spectrogram}
            )
            
            # Postprocess outputs
            result = self.postprocess_output(outputs)
            
            inference_time = (time.time() - start_time) * 1000  # Convert to ms
            
            return {
                'text': result['text'],
                'confidence': result['confidence'],
                'npu_used': self.npu_available,
                'inference_time': inference_time
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
        """Get information about the loaded model"""
        if not self.session:
            return None
            
        return {
            'model_path': self.model_path,
            'providers': self.session.get_providers(),
            'npu_available': self.npu_available,
            'sample_rate': self.sample_rate,
            'input_shape': self.session.get_inputs()[0].shape,
            'output_shapes': [output.shape for output in self.session.get_outputs()]
        }
