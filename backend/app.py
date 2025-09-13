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
    
    try:
        logger.info("Initializing OCR engine with QNN...")
        detector_model_path = "../models/easyocr-easyocrdetector.onnx/model.onnx/model.onnx"
        recognizer_model_path = "../models/easyocr-easyocrrecognizer.onnx/model.onnx/model.onnx"
        ocr_engine = OCREngine(detector_model_path=detector_model_path, recognizer_model_path=recognizer_model_path)
    except Exception as ocr_error:
        logger.error(f"Failed to initialize OCR engine: {ocr_error}. OCR functions will be unavailable.")
        ocr_engine = None

    try:
        logger.info("Initializing Whisper engine with QNN...")
        whisper_engine = WhisperEngine()
    except Exception as whisper_error:
        logger.error(f"Failed to initialize Whisper engine: {whisper_error}. Whisper functions will be unavailable.")
        whisper_engine = None

    try:
        logger.info("Initializing command agent...")
        command_agent = CommandAgent()
    except Exception as agent_error:
        logger.error(f"Failed to initialize Command agent: {agent_error}. Agent functions will be unavailable.")
        command_agent = None
    
    if ocr_engine is not None or whisper_engine is not None or command_agent is not None:
        logger.info("Some engines initialized successfully!")
        return True
    else:
        logger.error("All engines failed to initialize.")
        return False

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for offline status verification"""
    return jsonify({
        'status': 'online',
        'offline': True,
        'engines': {
            'ocr': ocr_engine is not None,
            'whisper': whisper_engine is not None,
            'agent': command_agent is not None
        },
        'npu_available': ocr_engine.npu_available if ocr_engine else False
    })

@app.route('/ocr', methods=['POST'])
def process_ocr():
    """Process webcam image for text extraction using EasyOCR + QNN"""
    try:
        if not ocr_engine:
            return jsonify({'error': 'OCR engine not initialized'}), 500
            
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
        
        return jsonify({
            'success': True,
            'text': result['text'],
            'confidence': result['confidence'],
            'npu_used': result['npu_used'],
            'inference_time': result['inference_time']
        })
        
    except Exception as e:
        logger.error(f"OCR processing error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/stt', methods=['POST'])
def process_speech():
    """Process audio for speech-to-text using Whisper + QNN"""
    try:
        if not whisper_engine:
            return jsonify({'error': 'Whisper engine not initialized'}), 500
            
        # Get audio data from request
        data = request.get_json()
        if 'audio' not in data:
            return jsonify({'error': 'No audio data provided'}), 400
            
        # Decode base64 audio
        audio_data = data['audio'].split(',')[1]  # Remove data:audio/wav;base64, prefix
        audio_bytes = base64.b64decode(audio_data)
        
        # Process with Whisper engine
        result = whisper_engine.transcribe(audio_bytes)
        
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
