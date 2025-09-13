"""
iLumina Backend - Flask API for offline exam assistance
Optimized for Qualcomm NPU (QNN Execution Provider)
"""

import os
import sys
import json
import base64
import io
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
from PIL import Image

# Import our custom modules
from ocr_qnn import OCREngine
from whisper_qnn import WhisperEngine
from agent_local import CommandAgent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Initialize engines
ocr_engine = None
whisper_engine = None
command_agent = None

def initialize_engines():
    """Initialize all AI engines with QNN Execution Provider"""
    global ocr_engine, whisper_engine, command_agent
    
    engines_initialized = 0
    total_engines = 3
    
    try:
        logger.info("Initializing OCR engine with QNN...")
        ocr_engine = OCREngine(
            detector_path="../models/easyocr_detector/model.onnx",
            recognizer_path="../models/easyocr_recognizer/model.onnx"
        )
        if ocr_engine.is_ready():
            engines_initialized += 1
            logger.info("✓ OCR engine initialized successfully")
        else:
            logger.warning("⚠ OCR engine initialized in limited mode (requires QNN/NPU)")
        
    except Exception as e:
        logger.error(f"✗ Failed to initialize OCR engine: {e}")
        ocr_engine = None
    
    try:
        logger.info("Initializing Whisper engine with QNN...")
        whisper_engine = WhisperEngine(
            encoder_path="../models/whisper_encoder/model.onnx",
            decoder_path="../models/whisper_decoder/model.onnx"
        )
        if whisper_engine.is_ready():
            engines_initialized += 1
            logger.info("✓ Whisper engine initialized successfully")
        else:
            logger.warning("⚠ Whisper engine initialized in limited mode (requires QNN/NPU)")
        
    except Exception as e:
        logger.error(f"✗ Failed to initialize Whisper engine: {e}")
        whisper_engine = None
    
    try:
        logger.info("Initializing command agent...")
        command_agent = CommandAgent()
        engines_initialized += 1
        logger.info("✓ Command agent initialized successfully")
        
    except Exception as e:
        logger.error(f"✗ Failed to initialize command agent: {e}")
        command_agent = None
    
    logger.info(f"Engine initialization complete: {engines_initialized}/{total_engines} engines ready")
    
    # Allow app to start if at least OCR is working
    if ocr_engine is not None:
        return True
    else:
        logger.error("Critical failure: OCR engine is required but failed to initialize")
        return False

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for offline status verification"""
    return jsonify({
        'status': 'online',
        'offline': True,
        'engines': {
            'ocr': ocr_engine is not None and ocr_engine.is_ready(),
            'whisper': whisper_engine is not None and whisper_engine.is_ready(),
            'agent': command_agent is not None
        },
        'npu_available': ocr_engine.npu_available if ocr_engine else False,
        'ocr_ready': ocr_engine.is_ready() if ocr_engine else False,
        'whisper_ready': whisper_engine.is_ready() if whisper_engine else False
    })

@app.route('/ocr', methods=['POST'])
def process_ocr():
    """Process webcam image for text extraction using EasyOCR + QNN"""
    try:
        if not ocr_engine:
            return jsonify({
                'success': False,
                'error': 'OCR engine not initialized',
                'text': '',
                'confidence': 0.0,
                'npu_used': False,
                'inference_time': 0.0
            }), 503
            
        if not ocr_engine.is_ready():
            return jsonify({
                'success': False,
                'error': 'OCR engine not ready - requires QNN/NPU hardware',
                'text': '',
                'confidence': 0.0,
                'npu_used': False,
                'inference_time': 0.0
            }), 503
            
        # Get image data from request
        data = request.get_json()
        if 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400
            
        # Decode base64 image
        image_data = data['image'].split(',')[1]  # Remove data:image/jpeg;base64, prefix
        image_bytes = base64.b64decode(image_data)
        
        # Convert to PIL Image
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to OpenCV format
        opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Process with OCR engine
        result = ocr_engine.extract_text(opencv_image)
        
        # Check if OCR was successful
        if result.get('error'):
            return jsonify({
                'success': False,
                'error': result['error'],
                'text': result.get('text', ''),
                'confidence': result.get('confidence', 0.0),
                'npu_used': result.get('npu_used', False),
                'inference_time': result.get('inference_time', 0.0)
            }), 500
        
        return jsonify({
            'success': True,
            'text': result['text'],
            'confidence': result['confidence'],
            'npu_used': result['npu_used'],
            'inference_time': result['inference_time'],
            'regions_detected': result.get('regions_detected', 0)
        })
        
    except Exception as e:
        logger.error(f"OCR processing error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/stt', methods=['POST'])
def process_speech():
    """Process audio for speech-to-text using Whisper + QNN"""
    try:
        if not whisper_engine:
            return jsonify({
                'success': False,
                'error': 'Whisper engine not initialized',
                'text': '',
                'confidence': 0.0,
                'npu_used': False,
                'inference_time': 0.0
            }), 503
            
        if not whisper_engine.is_ready():
            return jsonify({
                'success': False,
                'error': 'Whisper engine not ready - requires QNN/NPU hardware',
                'text': '',
                'confidence': 0.0,
                'npu_used': False,
                'inference_time': 0.0
            }), 503
            
        # Get audio data from request
        data = request.get_json()
        if 'audio' not in data:
            return jsonify({'error': 'No audio data provided'}), 400
            
        # Decode base64 audio
        audio_data = data['audio'].split(',')[1]  # Remove data:audio/wav;base64, prefix
        audio_bytes = base64.b64decode(audio_data)
        
        # Process with Whisper engine
        result = whisper_engine.transcribe(audio_bytes)
        
        # Check if transcription was successful
        if result.get('error'):
            return jsonify({
                'success': False,
                'error': result['error'],
                'text': result.get('text', ''),
                'confidence': result.get('confidence', 0.0),
                'npu_used': result.get('npu_used', False),
                'inference_time': result.get('inference_time', 0.0)
            }), 500
        
        return jsonify({
            'success': True,
            'text': result['text'],
            'confidence': result['confidence'],
            'npu_used': result['npu_used'],
            'inference_time': result['inference_time']
        })
        
    except Exception as e:
        logger.error(f"Speech processing error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/command', methods=['POST'])
def process_command():
    """Process text commands and execute actions"""
    try:
        if not command_agent:
            return jsonify({'error': 'Command agent not initialized'}), 500
            
        data = request.get_json()
        if 'text' not in data:
            return jsonify({'error': 'No command text provided'}), 400
            
        command_text = data['text']
        
        # Process command with agent
        result = command_agent.process_command(command_text)
        
        return jsonify({
            'success': True,
            'action': result['action'],
            'response': result['response'],
            'tts_text': result['tts_text']
        })
        
    except Exception as e:
        logger.error(f"Command processing error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/tts', methods=['POST'])
def text_to_speech():
    """Convert text to speech using offline TTS"""
    try:
        if not command_agent:
            return jsonify({'error': 'Command agent not initialized'}), 500
            
        data = request.get_json()
        if 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400
            
        text = data['text']
        
        # Use command agent's TTS
        command_agent.speak(text)
        
        return jsonify({
            'success': True,
            'message': 'Speech generated successfully'
        })
        
    except Exception as e:
        logger.error(f"TTS error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Initialize engines on startup
    if initialize_engines():
        logger.info("Starting iLumina backend server...")
        app.run(host='127.0.0.1', port=5000, debug=False)
    else:
        logger.error("Failed to initialize engines. Exiting...")
        sys.exit(1)
