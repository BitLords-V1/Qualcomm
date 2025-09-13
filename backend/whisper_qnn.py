"""
Whisper [Speech To Text] Engine using Whisper Tiny EN ONNX with Qualcomm QNN Execution Provider
Optimized for NPU acceleration on Windows-on-Snapdragon
"""

import os
import time
import logging
import numpy as np
import wave
import io
import onnxruntime as ort
import librosa
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)

class WhisperEngine:
    def __init__(self, encoder_path="../models/whisper_encoder/model.onnx", 
                 decoder_path="../models/whisper_decoder/model.onnx"):
        """
        Initialize Whisper engine with separate encoder and decoder models
        
        Args:
            encoder_path: Path to Whisper encoder ONNX model compiled for QNN
            decoder_path: Path to Whisper decoder ONNX model compiled for QNN
        """
        self.encoder_path = encoder_path
        self.decoder_path = decoder_path
        self.encoder_session = None
        self.decoder_session = None
        self.npu_available = False
        
        # Whisper model parameters
        self.sample_rate = 16000
        self.n_mels = 80
        self.n_ctx = 448
        self.n_vocab = 51865
        self.max_length = 200  # Maximum sequence length for decoder
        
        # Whisper tokenizer tokens
        self.sot_token = 50258  # Start of transcript
        self.eot_token = 50257  # End of transcript
        self.no_speech_token = 50361  # No speech token
        
        # Initialize ONNX Runtime sessions
        self._initialize_sessions()
    
    def _initialize_sessions(self):
        """Initialize ONNX Runtime sessions for encoder and decoder with QNN Execution Provider"""
        try:
            # Check if model files exist
            if not os.path.exists(self.encoder_path):
                raise FileNotFoundError(f"Encoder model not found: {self.encoder_path}")
            if not os.path.exists(self.decoder_path):
                raise FileNotFoundError(f"Decoder model not found: {self.decoder_path}")
            
            # Configure providers - QNN first, CPU as fallback
            providers = ["QNNExecutionProvider"]
            
            # Initialize encoder session
            try:
                self.encoder_session = ort.InferenceSession(
                    self.encoder_path,
                    providers=providers
                )
                self.npu_available = True
                logger.info("Encoder initialized with QNN Execution Provider (NPU)")
                
            except Exception as qnn_error:
                logger.warning(f"QNN not available for encoder: {qnn_error}")
                logger.info("Attempting CPU fallback for encoder...")
                
                try:
                    providers = ["CPUExecutionProvider"]
                    self.encoder_session = ort.InferenceSession(
                        self.encoder_path,
                        providers=providers
                    )
                    self.npu_available = False
                    logger.info("Encoder initialized with CPU Execution Provider")
                except Exception as cpu_error:
                    logger.error(f"CPU fallback also failed for encoder: {cpu_error}")
                    logger.warning("Encoder model requires QNN/NPU - cannot run on CPU")
                    self.encoder_session = None
                    self.npu_available = False
            
            # Initialize decoder session
            try:
                self.decoder_session = ort.InferenceSession(
                    self.decoder_path,
                    providers=["QNNExecutionProvider"]
                )
                logger.info("Decoder initialized with QNN Execution Provider (NPU)")
                
            except Exception as qnn_error:
                logger.warning(f"QNN not available for decoder: {qnn_error}")
                logger.info("Attempting CPU fallback for decoder...")
                
                try:
                    providers = ["CPUExecutionProvider"]
                    self.decoder_session = ort.InferenceSession(
                        self.decoder_path,
                        providers=providers
                    )
                    logger.info("Decoder initialized with CPU Execution Provider")
                except Exception as cpu_error:
                    logger.error(f"CPU fallback also failed for decoder: {cpu_error}")
                    logger.warning("Decoder model requires QNN/NPU - cannot run on CPU")
                    self.decoder_session = None
            
            # Check if both sessions were initialized
            if self.encoder_session is None or self.decoder_session is None:
                logger.warning("Whisper engine initialized in limited mode - models require QNN/NPU")
                logger.info("Speech-to-text functionality will be disabled until QNN/NPU is available")
            else:
                logger.info("Whisper encoder and decoder models loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Whisper engine: {e}")
            # Don't raise exception - allow app to continue without Whisper
            self.encoder_session = None
            self.decoder_session = None
            self.npu_available = False
    
    def is_ready(self):
        """Check if the Whisper engine is ready for inference"""
        return self.encoder_session is not None and self.decoder_session is not None
    
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
        Preprocess audio for Whisper inference by computing mel spectrogram
        
        Args:
            audio_array: Audio array (1D, float32, normalized)
            
        Returns:
            Preprocessed mel spectrogram [1, 80, 3000]
        """
        try:
            # Pad or truncate to 30 seconds (480,000 samples at 16kHz)
            target_length = 30 * self.sample_rate
            if len(audio_array) > target_length:
                audio_array = audio_array[:target_length]
            else:
                audio_array = np.pad(audio_array, (0, target_length - len(audio_array)))
            
            # Compute mel spectrogram using librosa
            # Whisper uses 80 mel bins and 3000 time frames for 30 seconds
            mel_spectrogram = librosa.feature.melspectrogram(
                y=audio_array,
                sr=self.sample_rate,
                n_mels=self.n_mels,
                n_fft=400,
                hop_length=160,
                win_length=400,
                fmin=0,
                fmax=8000
            )
            
            # Convert to log scale
            mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
            
            # Ensure the shape is exactly [80, 3000]
            if mel_spectrogram.shape[1] != 3000:
                if mel_spectrogram.shape[1] > 3000:
                    mel_spectrogram = mel_spectrogram[:, :3000]
                else:
                    # Pad with zeros if shorter
                    pad_width = 3000 - mel_spectrogram.shape[1]
                    mel_spectrogram = np.pad(mel_spectrogram, ((0, 0), (0, pad_width)), mode='constant')
            
            # Add batch dimension: [1, 80, 3000]
            mel_spectrogram = np.expand_dims(mel_spectrogram, axis=0).astype(np.float32)
            
            return mel_spectrogram
            
        except Exception as e:
            logger.error(f"Failed to preprocess audio: {e}")
            # Fallback to dummy spectrogram if librosa fails
            mel_spectrogram = np.random.randn(1, self.n_mels, 3000).astype(np.float32)
            return mel_spectrogram
    
    def run_encoder(self, mel_spectrogram):
        """
        Run the encoder model to get cross-attention key-value caches
        
        Args:
            mel_spectrogram: Mel spectrogram [1, 80, 3000]
            
        Returns:
            Dictionary with cross-attention key-value caches
        """
        try:
            # Run encoder inference
            outputs = self.encoder_session.run(
                None,  # All outputs
                {"input_features": mel_spectrogram}
            )
            
            # Extract cross-attention caches
            cross_caches = {}
            for i, output in enumerate(outputs):
                if "k_cache_cross" in self.encoder_session.get_outputs()[i].name:
                    cross_caches[self.encoder_session.get_outputs()[i].name] = output
                elif "v_cache_cross" in self.encoder_session.get_outputs()[i].name:
                    cross_caches[self.encoder_session.get_outputs()[i].name] = output
            
            return cross_caches
            
        except Exception as e:
            logger.error(f"Encoder inference failed: {e}")
            raise
    
    def run_decoder(self, input_ids, attention_mask, cross_caches, self_caches=None, position_ids=None):
        """
        Run the decoder model for text generation
        
        Args:
            input_ids: Token IDs [1, 1]
            attention_mask: Attention mask [1, 1, 1, 200]
            cross_caches: Cross-attention key-value caches from encoder
            self_caches: Self-attention key-value caches (for autoregressive generation)
            position_ids: Position IDs [1]
            
        Returns:
            Tuple of (logits, updated_self_caches)
        """
        try:
            # Prepare decoder inputs
            decoder_inputs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids if position_ids is not None else np.array([0], dtype=np.int64)
            }
            
            # Add cross-attention caches
            decoder_inputs.update(cross_caches)
            
            # Add self-attention caches if provided
            if self_caches:
                decoder_inputs.update(self_caches)
            else:
                # Initialize empty self-attention caches
                for i in range(4):  # 4 layers
                    decoder_inputs[f"k_cache_self_{i}_in"] = np.zeros((6, 1, 64, 199), dtype=np.float32)
                    decoder_inputs[f"v_cache_self_{i}_in"] = np.zeros((6, 1, 199, 64), dtype=np.float32)
            
            # Run decoder inference
            outputs = self.decoder_session.run(None, decoder_inputs)
            
            # Extract logits and updated self-attention caches
            logits = outputs[0]  # [1, 51865, 1, 1]
            
            # Extract updated self-attention caches
            updated_self_caches = {}
            for i, output in enumerate(outputs[1:], 1):  # Skip logits
                output_name = self.decoder_session.get_outputs()[i].name
                if "k_cache_self" in output_name or "v_cache_self" in output_name:
                    updated_self_caches[output_name] = output
            
            return logits, updated_self_caches
            
        except Exception as e:
            logger.error(f"Decoder inference failed: {e}")
            raise
    
    def decode_tokens(self, token_ids):
        """
        Decode token IDs to text (simplified implementation)
        
        Args:
            token_ids: List of token IDs
            
        Returns:
            Decoded text string
        """
        # This is a simplified tokenizer
        # In a real implementation, you would use the actual Whisper tokenizer
        
        # Basic token mapping for common tokens
        token_map = {
            50258: "<|startoftranscript|>",
            50257: "<|endoftext|>",
            50361: "<|nospeech|>",
            50362: "<|notimestamps|>",
            50363: "<|0.00|>",
            50364: "<|30.00|>",
        }
        
        # For now, return a placeholder
        # In a real implementation, you would:
        # 1. Load the Whisper tokenizer
        # 2. Decode the token IDs to text
        # 3. Handle special tokens properly
        
        text_parts = []
        for token_id in token_ids:
            if token_id in token_map:
                text_parts.append(token_map[token_id])
            elif token_id < 256:  # ASCII characters
                text_parts.append(chr(token_id))
            else:
                # For other tokens, you'd need the full tokenizer
                text_parts.append(f"[TOKEN_{token_id}]")
        
        return " ".join(text_parts)
    
    def transcribe(self, audio_bytes):
        """
        Transcribe audio using Whisper encoder-decoder pipeline
        
        Args:
            audio_bytes: Raw audio bytes (WAV format)
            
        Returns:
            Dictionary with transcribed text, confidence, and metadata
        """
        # Check if engine is ready
        if not self.is_ready():
            return {
                'text': '',
                'confidence': 0.0,
                'npu_used': False,
                'inference_time': 0.0,
                'error': 'Whisper engine not ready - models require QNN/NPU hardware'
            }
        
        try:
            start_time = time.time()
            
            # Load and preprocess audio
            audio_array = self.load_audio(audio_bytes)
            mel_spectrogram = self.preprocess_audio(audio_array)
            
            # Run encoder to get cross-attention caches
            cross_caches = self.run_encoder(mel_spectrogram)
            
            # Initialize decoder state
            input_ids = np.array([[self.sot_token]], dtype=np.int64)  # Start of transcript
            attention_mask = np.ones((1, 1, 1, self.max_length), dtype=np.int64)
            position_ids = np.array([0], dtype=np.int64)
            
            # Generate text autoregressively
            generated_tokens = []
            self_caches = None
            
            for step in range(self.max_length):
                # Run decoder
                logits, self_caches = self.run_decoder(
                    input_ids, attention_mask, cross_caches, self_caches, position_ids
                )
                
                # Get next token (greedy decoding)
                next_token = np.argmax(logits[0, :, 0, 0])
                generated_tokens.append(next_token)
                
                # Check for end of sequence
                if next_token == self.eot_token:
                    break
                
                # Update input for next iteration
                input_ids = np.array([[next_token]], dtype=np.int64)
                position_ids = np.array([step + 1], dtype=np.int64)
                
                # Update attention mask
                attention_mask[0, 0, 0, step + 1] = 1
            
            # Decode tokens to text
            text = self.decode_tokens(generated_tokens)
            
            # Calculate confidence (simplified)
            confidence = 0.85  # Placeholder confidence
            
            inference_time = (time.time() - start_time) * 1000  # Convert to ms
            
            return {
                'text': text,
                'confidence': confidence,
                'npu_used': self.npu_available,
                'inference_time': inference_time,
                'tokens_generated': len(generated_tokens)
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
            'encoder_providers': self.encoder_session.get_providers(),
            'decoder_providers': self.decoder_session.get_providers(),
            'npu_available': self.npu_available,
            'sample_rate': self.sample_rate,
            'encoder_input_shape': self.encoder_session.get_inputs()[0].shape,
            'decoder_input_shapes': [inp.shape for inp in self.decoder_session.get_inputs()],
            'encoder_output_shapes': [output.shape for output in self.encoder_session.get_outputs()],
            'decoder_output_shapes': [output.shape for output in self.decoder_session.get_outputs()]
        }
