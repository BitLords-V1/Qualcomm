"""
iLumina Backend - Simple PDF OCR for MCQ Questions
"""

import os
import sys
import json
import base64
import io
import logging
import ssl
from flask import Flask, request, jsonify
from flask_cors import CORS
import fitz  # PyMuPDF
import pytesseract
import re
import numpy as np
from PIL import Image
from tts_engine import tts_engine

# Fix SSL certificate issues
ssl._create_default_https_context = ssl._create_unverified_context

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Configure Tesseract path (if needed)
# pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract'

def initialize_ocr():
    """Initialize Tesseract OCR"""
    try:
        logger.info("Initializing Tesseract OCR...")
        # Test Tesseract by running a simple command
        pytesseract.get_tesseract_version()
        logger.info("✅ Tesseract OCR initialized successfully!")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to initialize Tesseract: {e}")
        return False

def pdf_to_images(pdf_path):
    """Convert PDF pages to images"""
    try:
        doc = fitz.open(pdf_path)
        images = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            # Convert to image with good quality
            mat = fitz.Matrix(2.0, 2.0)  # 2x zoom
            pix = page.get_pixmap(matrix=mat)
            img_data = pix.tobytes("png")
            images.append(img_data)
            logger.info(f"Converted page {page_num + 1}/{len(doc)}")
        
        doc.close()
        return images
    except Exception as e:
        logger.error(f"PDF conversion failed: {e}")
        return []

def extract_text_from_image(image_data):
    """Extract text from image using Tesseract OCR"""
    try:
        # Convert bytes to PIL Image
        img = Image.open(io.BytesIO(image_data))
        
        # Run Tesseract OCR
        # Use PSM 6 for uniform block of text (good for documents)
        text = pytesseract.image_to_string(img, config='--psm 6')
        
        # Clean up the text
        text = text.strip()
        
        if text:
            logger.info(f"Extracted text: {text[:200]}...")
            return text
        else:
            logger.warning("No text extracted from image")
            return ""
        
    except Exception as e:
        logger.error(f"Tesseract OCR failed: {e}")
        return ""

def format_as_questions(extracted_text):
    """Format extracted text as properly structured MCQ questions with options"""
    try:
        if not extracted_text:
            return "No text detected"
        
        # Clean up text but preserve some structure
        combined_text = re.sub(r'\s+', ' ', extracted_text).strip()
        
        logger.info(f"Combined text: {combined_text[:300]}...")
        
        # Remove header/instructions (text before first question number)
        first_question_match = re.search(r'\b1[\.\)]\s*', combined_text)
        if first_question_match:
            combined_text = combined_text[first_question_match.start():]
            logger.info(f"Removed header, text now starts with: {combined_text[:100]}...")
        
        # Simple but effective approach: Split by question numbers at word boundaries
        # This prevents matching numbers in the middle of sentences
        question_pattern = r'(\b\d+[\.\)]\s+)'
        parts = re.split(question_pattern, combined_text)
        
        logger.info(f"Split into {len(parts)} parts")
        
        questions = []
        
        # Process each part
        for i in range(1, len(parts), 2):  # Skip question numbers, process content
            if i + 1 < len(parts):
                question_number = parts[i].strip()
                question_content = parts[i + 1].strip()
                
                logger.info(f"Processing question {question_number}: {question_content[:100]}...")
                
                if question_content:
                    # Clean up the question content
                    question_content = re.sub(r'\s+', ' ', question_content).strip()
                    
                    # Add the question number back
                    full_question = question_number + question_content
                    
                    # Ensure it ends with proper punctuation
                    if not full_question.endswith(('?', '.', '!')):
                        full_question += "?"
                    
                    questions.append(full_question)
        
        # Format as Q1, Q2, Q3...
        formatted_questions = []
        for i, question in enumerate(questions, 1):
            formatted_questions.append(f"Q{i}: {question}")
        
        logger.info(f"Found {len(formatted_questions)} questions")
        return "\n\n".join(formatted_questions)
        
    except Exception as e:
        logger.error(f"Question formatting failed: {e}")
        return "Error formatting questions"


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'ocr_ready': True,  # Tesseract is always ready
        'tts_ready': tts_engine.get_status()['available']
    })

@app.route('/speak', methods=['POST'])
def speak_endpoint():
    """Speak text using TTS"""
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        if not text.strip():
            return jsonify({'error': 'No text provided'}), 400
        
        tts_engine.speak(text)
        
        return jsonify({
            'success': True,
            'message': 'Text is being spoken',
            'text': text
        })
        
    except Exception as e:
        logger.error(f"TTS endpoint error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/repeat', methods=['POST'])
def repeat_text():
    """Repeat the last extracted text"""
    try:
        tts_engine.repeat()
        
        return jsonify({
            'success': True,
            'message': 'Repeating last text',
            'text': tts_engine.current_text
        })
        
    except Exception as e:
        logger.error(f"Repeat endpoint error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/upload-pdf', methods=['POST'])
def upload_pdf():
    """Handle PDF upload and OCR processing"""
    try:
        
        # Check if file was uploaded
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not file.filename.lower().endswith('.pdf'):
            return jsonify({'error': 'File must be a PDF'}), 400
        
        # Save file temporarily
        temp_path = f"/tmp/{file.filename}"
        file.save(temp_path)
        
        try:
            logger.info(f"Processing PDF: {file.filename}")
            
            # Convert PDF to images
            images = pdf_to_images(temp_path)
            if not images:
                return jsonify({'error': 'Failed to convert PDF to images'}), 500
            
            # Extract text from each page
            all_text = ""
            for i, image_data in enumerate(images):
                logger.info(f"Processing page {i + 1}/{len(images)}")
                page_text = extract_text_from_image(image_data)
                if page_text:
                    all_text += page_text + "\n"
            
            # Format as questions
            questions = format_as_questions(all_text)
            
            # Speak the extracted text
            if all_text.strip():
                logger.info(f"Extracted text from PDF: {all_text[:100]}...")
                logger.info("Calling TTS engine for PDF...")
                tts_engine.speak(all_text)
                logger.info("TTS call completed for PDF")
            else:
                logger.warning("No text extracted from PDF, skipping TTS")
            
            return jsonify({
                'success': True,
                'questions': questions,
                'pages_processed': len(images),
                'filename': file.filename,
                'extracted_text': all_text,
                'tts_spoken': bool(all_text.strip())
            })
            
        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    except Exception as e:
        logger.error(f"PDF processing error: {e}")
@app.route('/upload-image', methods=['POST'])
def upload_image():
    """Handle image upload and OCR processing"""
    try:
        
        # Check if file was uploaded
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Check if it's an image
        allowed_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff'}
        file_ext = os.path.splitext(file.filename.lower())[1]
        if file_ext not in allowed_extensions:
            return jsonify({'error': 'File must be an image (PNG, JPG, JPEG, GIF, BMP, TIFF)'}), 400
        
        # Save file temporarily
        temp_path = f"/tmp/{file.filename}"
        file.save(temp_path)
        
        try:
            logger.info(f"Processing image: {file.filename}")
            
            # Read image file
            with open(temp_path, 'rb') as f:
                image_data = f.read()
            
            # Extract text from image
            extracted_text = extract_text_from_image(image_data)
            
            # Format as questions
            questions = format_as_questions(extracted_text)
            
            # Speak the extracted text
            if extracted_text.strip():
                logger.info(f"Extracted text: {extracted_text[:100]}...")
                logger.info("Calling TTS engine...")
                tts_engine.speak(extracted_text)
                logger.info("TTS call completed")
            else:
                logger.warning("No text extracted, skipping TTS")
            
            return jsonify({
                'success': True,
                'questions': questions,
                'filename': file.filename,
                'extracted_text': extracted_text,
                'tts_spoken': bool(extracted_text.strip())
            })
            
        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    except Exception as e:
        logger.error(f"Image processing error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Initialize OCR
    if initialize_ocr():
        logger.info("🚀 Starting PDF OCR server...")
    else:
        logger.error("❌ Failed to initialize OCR. Exiting...")
        sys.exit(1)
    
    # TTS is automatically initialized when imported
    tts_status = tts_engine.get_status()
    if tts_status['available']:
        logger.info(f"🎤 TTS engine ready! ({tts_status['engine']})")
    else:
        logger.warning("⚠️ TTS not available - speech features disabled")
    
    # Start Flask server
    app.run(host='127.0.0.1', port=5000, debug=True)
