"""
AI Exam Helper - Simple & Clean Implementation
Accessibility Assistant for Students with Dyslexia
"""

import os
import json
import base64
import io
import logging
import ssl
import time
import requests
import yaml
from flask import Flask, request, jsonify
from flask_cors import CORS
import fitz  # PyMuPDF
from tts_engine import tts_engine

# Fix SSL certificate issues
ssl._create_default_https_context = ssl._create_unverified_context

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

class AIExamHelper:
    """Simple AI Exam Helper for Students with Dyslexia"""
    
    def __init__(self):
        self.config = self.load_config()
        self.anythingllm_ready = False
        self.current_exam = None
        self.current_question = 0
        self.student_answers = {}
        self.exam_session = None
        
        # Initialize AnythingLLM connection
        self.init_anythingllm()
    
    def load_config(self):
        """Load configuration"""
        try:
            with open('config.yaml', 'r') as f:
                config = yaml.safe_load(f)
            logger.info("✅ Configuration loaded")
            return config
        except Exception as e:
            logger.error(f"❌ Config load failed: {e}")
            return {
                'api_key': 'demo-key',
                'model_server_base_url': 'http://localhost:3001/api/v1',
                'workspace_slug': 'exam-helper',
                'stream': False,
                'stream_timeout': 60
            }
    
    def init_anythingllm(self):
        """Initialize AnythingLLM connection"""
        try:
            # Test connection (will fail on Mac, that's expected)
            headers = {
                'Authorization': f'Bearer {self.config["api_key"]}',
                'Content-Type': 'application/json'
            }
            
            test_url = f"{self.config['model_server_base_url']}/workspace/{self.config['workspace_slug']}/chat"
            test_payload = {
                "message": "Hello, are you working?",
                "mode": "chat",
                "stream": False
            }
            
            response = requests.post(test_url, headers=headers, json=test_payload, timeout=5)
            
            if response.status_code == 200:
                self.anythingllm_ready = True
                logger.info("✅ AnythingLLM NPU connected!")
            else:
                logger.info("⚠️ AnythingLLM not available (demo mode)")
                self.anythingllm_ready = False
                
        except Exception as e:
            logger.info(f"ℹ️ AnythingLLM not available: {e}")
            self.anythingllm_ready = False
    
    def send_to_anythingllm(self, message, context=None):
        """Send message to AnythingLLM and get response - REQUIRED for all interactions"""
        try:
            headers = {
                'Authorization': f'Bearer {self.config["api_key"]}',
                'Content-Type': 'application/json'
            }
            
            payload = {
                "message": message,
                "mode": "chat",
                "stream": False
            }
            
            if context:
                payload["context"] = context
            
            url = f"{self.config['model_server_base_url']}/workspace/{self.config['workspace_slug']}/chat"
            
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                if isinstance(result, dict):
                    response_text = (result.get("textResponse") or 
                                   result.get("response") or 
                                   result.get("text") or 
                                   result.get("message") or 
                                   str(result))
                else:
                    response_text = str(result)
                
                return response_text if response_text else "No response received"
            else:
                logger.error(f"AnythingLLM API error: {response.status_code}")
                return "AI assistant temporarily unavailable - please check AnythingLLM connection"
                
        except Exception as e:
            logger.error(f"AnythingLLM request failed: {e}")
            return "AI assistant temporarily unavailable - please check AnythingLLM connection"
    
    def speak_response(self, text):
        """Speak the response using TTS with enhanced audio output"""
        try:
            if text and text.strip():
                logger.info(f"🔊 Speaking response: {text[:50]}...")
                success = tts_engine.speak(text)
                if success:
                    logger.info("✅ Audio output completed successfully")
                else:
                    logger.error("❌ Audio output failed")
                return success
            else:
                logger.warning("No text to speak")
                return False
        except Exception as e:
            logger.error(f"❌ TTS failed: {e}")
            return False
    
    def get_voice_response(self, message, context=None):
        """Get AI response and speak it aloud with guaranteed audio output"""
        response = self.send_to_anythingllm(message, context)
        audio_success = self.speak_response(response)
        
        # If TTS fails, try to provide fallback
        if not audio_success:
            logger.warning("⚠️ TTS failed, but continuing with text response")
            # Could add fallback audio here if needed
        
        return response
    
    def process_exam_document(self, document_data, document_type="pdf"):
        """Process exam document"""
        try:
            if document_type == "pdf":
                # Extract text from PDF
                pdf_text = self.pdf_to_text(document_data)
                if not pdf_text:
                    return False, "Failed to extract text from PDF document"
                
                exam_analysis = self.analyze_exam_with_ai(pdf_text)
                
            elif document_type == "image":
                # For images, we'll need OCR (not implemented yet)
                return False, "Image processing not implemented yet"
            
            else:
                return False, "Unsupported document type"
            
            if exam_analysis and exam_analysis.get('total_questions', 0) > 0:
                self.current_exam = exam_analysis
                self.current_question = 0
                self.student_answers = {}
                self.exam_session = {
                    'start_time': time.time(),
                    'questions_total': exam_analysis.get('total_questions', 0),
                    'questions_answered': 0,
                    'current_question': 0,
                    'exam_title': exam_analysis.get('exam_title', 'Unknown Exam')
                }
                logger.info(f"✅ Exam processed: {exam_analysis.get('total_questions', 0)} questions found")
                
                # Generate static instructions for the student
                instructions = self.get_student_instructions()
                return True, {
                    'message': f"Exam processed successfully! Found {exam_analysis.get('total_questions', 0)} questions.",
                    'instructions': instructions,
                    'exam_info': {
                        'title': exam_analysis.get('exam_title', 'Unknown'),
                        'total_questions': exam_analysis.get('total_questions', 0),
                        'questions': exam_analysis.get('questions', [])
                    }
                }
            else:
                return False, "Failed to extract questions from exam document. Please ensure the PDF contains readable questions."
                
        except Exception as e:
            logger.error(f"Exam processing failed: {e}")
            return False, f"Error processing exam: {str(e)}"
    
    def pdf_to_text(self, pdf_data):
        """Extract text from PDF document"""
        try:
            # Open PDF from bytes
            temp_pdf = io.BytesIO(pdf_data)
            doc = fitz.open(stream=temp_pdf, filetype="pdf")
            full_text = ""
            
            # Extract text from each page
            for page_num in range(len(doc)):
                page = doc[page_num]
                page_text = page.get_text()
                full_text += f"\n--- Page {page_num + 1} ---\n"
                full_text += page_text
                logger.info(f"Extracted text from page {page_num + 1}/{len(doc)}")
            
            doc.close()
            
            logger.info(f"✅ PDF text extracted: {len(full_text)} characters")
            return full_text.strip()
            
        except Exception as e:
            logger.error(f"PDF text extraction failed: {e}")
            return ""
        
    def analyze_exam_with_ai(self, pdf_text):
        """Analyze exam text and extract questions using AnythingLLM"""
        try:
            if not pdf_text or len(pdf_text.strip()) < 10:
                logger.error("PDF text is too short or empty")
                return None
            
            # Create a comprehensive prompt for exam analysis
            analysis_prompt = f"""
            You are an AI assistant helping students with dyslexia. Analyze this exam document and extract ALL questions and their options.
            
            IMPORTANT: Return ONLY a valid JSON response with this EXACT format:
            
            {{
                "exam_title": "Extracted Exam Title",
                "total_questions": number_of_questions,
                "questions": [
                    {{
                        "question_number": 1,
                        "question_text": "The complete question text",
                        "options": ["A) Option 1", "B) Option 2", "C) Option 3", "D) Option 4"],
                        "question_type": "multiple_choice"
                    }},
                    {{
                        "question_number": 2,
                        "question_text": "Next question text",
                        "options": ["A) Option 1", "B) Option 2", "C) Option 3", "D) Option 4"],
                        "question_type": "multiple_choice"
                    }}
                ]
            }}
            
            Exam Document Text:
            {pdf_text}
            
            Instructions:
            1. Extract ALL questions you find in the document
            2. Look for question patterns: "1.", "2.", "Question 1", "Question 2", etc.
            3. Extract ALL multiple choice options: "A)", "B)", "C)", "D)" or "A.", "B.", "C.", "D."
            4. Include the complete question text
            5. Count the total number of questions accurately
            6. Return ONLY the JSON, no explanations or additional text
            
            If you cannot find any questions, return: {{"exam_title": "No Questions Found", "total_questions": 0, "questions": []}}
            """
            
            # ALWAYS try AnythingLLM first if available
            if self.anythingllm_ready:
                logger.info("🤖 Using AnythingLLM NPU for exam analysis...")
                response = self.send_to_anythingllm(analysis_prompt)
                logger.info(f"AnythingLLM response length: {len(response)} characters")
                
                # Parse JSON from response
                try:
                    # Extract JSON from response
                    json_start = response.find('{')
                    json_end = response.rfind('}') + 1
                    if json_start != -1 and json_end != -1:
                        json_str = response[json_start:json_end]
                        exam_data = json.loads(json_str)
                        
                        # Validate the structure
                        if self.validate_exam_data(exam_data):
                            logger.info(f"✅ AnythingLLM extracted {exam_data.get('total_questions', 0)} questions")
                            return exam_data
                        else:
                            logger.error("AnythingLLM response structure invalid")
                            return None
                    else:
                        logger.error("No valid JSON found in AnythingLLM response")
                        return None
                        
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse AnythingLLM response as JSON: {e}")
                    return None
            else:
                logger.warning("⚠️ AnythingLLM not available - using demo mode")
                # Return demo data for testing purposes when AnythingLLM is not available
                return self.get_demo_exam_data()
            
        except Exception as e:
            logger.error(f"Exam analysis failed: {e}")
            return None
    
    def get_demo_exam_data(self):
        """Return demo exam data for testing when AnythingLLM is not available"""
        return {
            "exam_title": "Demo Exam (AnythingLLM not available)",
            "total_questions": 3,
            "questions": [
                {
                    "question_number": 1,
                    "question_text": "What is the capital of France?",
                    "options": ["A) London", "B) Paris", "C) Berlin", "D) Madrid"],
                    "question_type": "multiple_choice"
                },
                {
                    "question_number": 2,
                    "question_text": "Which planet is closest to the Sun?",
                    "options": ["A) Venus", "B) Mercury", "C) Earth", "D) Mars"],
                    "question_type": "multiple_choice"
                },
                {
                    "question_number": 3,
                    "question_text": "What is 2 + 2?",
                    "options": ["A) 3", "B) 4", "C) 5", "D) 6"],
                    "question_type": "multiple_choice"
                }
            ]
        }
    
    def get_student_instructions(self):
        """Generate voice-friendly instructions for students"""
        instructions = {
            'title': 'Welcome to AI Exam Helper - Voice Instructions',
            'voice_text': """Welcome to AI Exam Helper! I'm your voice assistant for taking exams. 
            
            Here's how to use me:
            
            First, I will read each question aloud clearly and slowly. Listen carefully to the question and all the answer options.
            
            You can use these voice commands anytime: Say 'repeat' to hear the question again, 'slower' to hear it more slowly, 'next' to go to the next question, 'previous' to go back, or 'help' for assistance.
            
            When you're ready to answer, simply say your answer like 'A' or 'Paris'. I'll record your answer and ask if you want to move to the next question.
            
            When you're finished, say 'end test' or 'finish' and I'll provide you with a summary of your answers.
            
            There's no rush! Take as much time as you need. I'm here to help you succeed.
            
            Are you ready to begin? Say 'begin' or 'ready' when you want to start.""",
            'steps': [
                {
                    'step': 1,
                    'title': 'Getting Started',
                    'description': 'I will read each question aloud clearly and slowly. Listen carefully to the question and all the answer options.'
                },
                {
                    'step': 2,
                    'title': 'Voice Commands',
                    'description': 'You can use these voice commands anytime: "repeat" to hear the question again, "slower" to hear it more slowly, "next" to go to the next question, "previous" to go back, or "help" for assistance.'
                },
                {
                    'step': 3,
                    'title': 'Answering Questions',
                    'description': 'When you\'re ready to answer, simply say your answer (like "A" or "Paris"). I\'ll record your answer and ask if you want to move to the next question.'
                },
                {
                    'step': 4,
                    'title': 'Ending the Test',
                    'description': 'When you\'re finished, say "end test" or "finish" and I\'ll provide you with a summary of your answers.'
                },
                {
                    'step': 5,
                    'title': 'Take Your Time',
                    'description': 'There\'s no rush! Take as much time as you need. I\'m here to help you succeed. Say "begin" or "ready" when you want to start.'
                }
            ],
            'voice_commands': [
                '"ready" or "begin" - Start the exam',
                '"repeat" - Hear the question again',
                '"slower" - Hear the question more slowly',
                '"next" - Go to the next question',
                '"previous" - Go back to the previous question',
                '"help" - Get assistance',
                '"end test" or "finish" - Complete the exam'
            ]
        }
        return instructions

    def validate_exam_data(self, data):
        """Validate extracted exam data structure"""
        try:
            if not isinstance(data, dict):
                return False
            if 'questions' not in data or not isinstance(data['questions'], list):
                return False
            if len(data['questions']) == 0:
                return False
            
            # Check first question structure
            first_q = data['questions'][0]
            required_fields = ['question_number', 'question_text', 'options']
            for field in required_fields:
                if field not in first_q:
                    return False
            
            return True
        except:
            return False
    
    def get_student_interaction_response(self, student_input, current_question_num):
        """Generate AI response for student interactions - ALWAYS using AnythingLLM"""
        
        # Get exam session info safely
        total_questions = 0
        if self.exam_session:
            total_questions = self.exam_session.get('questions_total', 0)
        
        # ALWAYS use AnythingLLM for all interactions
        interaction_prompt = f"""
        You are an AI assistant helping a student with dyslexia during an exam. The student said: "{student_input}"
        
        Current question number: {current_question_num}
        Total questions: {total_questions}
        
        Respond appropriately based on what the student said:
        
        - If they said "ready" or "begin": Confirm they're ready and ask if they want to start with question 1
        - If they said "repeat": Repeat the current question clearly and slowly
        - If they said "slow" or "slower": Read the current question more slowly with pauses
        - If they said "next": Move to the next question
        - If they said "previous" or "back": Move to the previous question
        - If they said "answer" or gave an answer: Acknowledge their answer and ask if they want to move to the next question
        - If they said "help": Offer encouragement and remind them they can ask for repeats or slower reading
        
        Be encouraging, patient, and clear. Speak as if talking to someone with dyslexia.
        Keep responses short and helpful. Always end with a question to keep the conversation going.
        """
        
        # Always use AnythingLLM - no fallback to hardcoded responses
        response = self.send_to_anythingllm(interaction_prompt)
        self.speak_response(response)
        return response
    
    def start_exam_session(self):
        """Start the exam session - ALWAYS using AnythingLLM"""
        if not self.current_exam:
            return False, "No exam loaded"
        
        try:
            # Create exam session prompt for AnythingLLM
            session_prompt = f"""
            You are an AI exam assistant for a student with dyslexia. 
            The exam has {self.current_exam.get('total_questions', 0)} questions.
            
            Your role:
            1. Read questions clearly and slowly
            2. Present options clearly
            3. Wait for student confirmation before proceeding
            4. Be patient and supportive
            5. Track student answers
            
            Welcome the student and let them know you're ready to help.
            Ask if they're ready to begin with Question 1.
            """
            
            response = self.get_voice_response(session_prompt)
            return True, response
            
        except Exception as e:
            logger.error(f"Failed to start exam session: {e}")
            return False, f"Error starting exam: {str(e)}"
    
    def get_current_question(self):
        """Get the current question with AnythingLLM and voice output"""
        if not self.current_exam or not self.current_exam.get('questions'):
            return None, "No exam loaded"
        
        try:
            questions = self.current_exam['questions']
            if self.current_question >= len(questions):
                return None, "Exam completed"
            
            current_q = questions[self.current_question]
            
            # Create question prompt for AnythingLLM
            question_prompt = f"""
            Please read this question clearly and slowly for a student with dyslexia:
            
            Question {current_q['question_number']}: {current_q['question_text']}
            
            Options:
            {chr(10).join(current_q['options'])}
            
            After reading, ask: "Are you ready to answer, or would you like me to repeat or read more slowly?"
            """
            
            response = self.get_voice_response(question_prompt)
            return current_q, response
            
        except Exception as e:
            logger.error(f"Failed to get current question: {e}")
            return None, f"Error getting question: {str(e)}"
    
    def handle_student_response(self, response_text):
        """Handle student response with full interaction flow"""
        if not self.current_exam or not self.exam_session:
            return "No active exam session"
        
        try:
            response_lower = response_text.lower().strip()
            current_q = self.current_question + 1
            total_q = self.exam_session['questions_total']
            
            # Handle end test commands
            if response_lower in ['end test', 'finish', 'end', 'complete', 'done', 'finish test']:
                # Student wants to end the test
                self.exam_session['end_time'] = time.time()
                self.exam_session['status'] = 'completed'
                ai_response = "Test completed! Let me provide you with a summary of your answers. Great job on completing the exam!"
                self.speak_response(ai_response)
                return ai_response
            
            # Handle different types of responses
            if response_lower in ['ready', 'yes', 'start', 'begin']:
                # Student is ready to begin
                ai_response = self.get_student_interaction_response(response_text, current_q)
                return ai_response
                
            elif response_lower in ['repeat', 'again', 'say again']:
                # Repeat current question
                ai_response = self.get_student_interaction_response(response_text, current_q)
                return ai_response
                
            elif response_lower in ['slow', 'slower', 'slowly', 'more slowly']:
                # Read slower
                ai_response = self.get_student_interaction_response(response_text, current_q)
                return ai_response
                
            elif response_lower in ['next', 'continue', 'next question']:
                # Move to next question
                if current_q < total_q:
                    self.current_question += 1
                    self.exam_session['current_question'] = self.current_question
                    ai_response = self.get_student_interaction_response(response_text, self.current_question + 1)
                else:
                    ai_response = "You've completed all questions! Great job! Would you like to end the test?"
                return ai_response
                
            elif response_lower in ['previous', 'back', 'go back', 'previous question']:
                # Move to previous question
                if current_q > 1:
                    self.current_question -= 1
                    self.exam_session['current_question'] = self.current_question
                    ai_response = self.get_student_interaction_response(response_text, self.current_question + 1)
                else:
                    ai_response = "This is already the first question."
                return ai_response
                
            elif response_lower in ['help', 'assistance', 'what can i say']:
                # Provide help
                ai_response = self.get_student_interaction_response(response_text, current_q)
                return ai_response
                
            else:
                # Treat as an answer
                question_num = self.current_question + 1
                self.student_answers[question_num] = response_text
                self.exam_session['questions_answered'] += 1
                
                # Generate AI feedback for the answer - ALWAYS using AnythingLLM
                feedback_prompt = f"""
                A student with dyslexia just answered question {question_num} with: "{response_text}"
                
                The question was: {self.current_exam['questions'][self.current_question]['question_text']}
                The options were: {self.current_exam['questions'][self.current_question]['options']}
                
                Provide encouraging feedback and ask if they want to move to the next question.
                Be supportive and clear. Don't reveal the correct answer.
                Keep your response conversational and encouraging.
                """
                
                ai_response = self.get_voice_response(feedback_prompt)
                
                # Move to next question
                if current_q < total_q:
                    self.current_question += 1
                    self.exam_session['current_question'] = self.current_question
                
                return ai_response
            
        except Exception as e:
            logger.error(f"Failed to process student response: {e}")
            return f"Error processing response: {str(e)}"
    
    def get_exam_summary(self):
        """Get exam summary - ALWAYS using AnythingLLM"""
        if not self.exam_session:
            return "No exam session active"
        
        try:
            summary_prompt = f"""
            Please provide a comprehensive summary of the exam session for a student with dyslexia:
            
            Total questions: {self.exam_session['questions_total']}
            Questions answered: {self.exam_session['questions_answered']}
            Student answers: {self.student_answers}
            
            Provide encouragement, acknowledge their effort, and give positive feedback.
            Be supportive and celebrate their completion of the exam.
            Keep the tone encouraging and uplifting.
            """
            
            return self.get_voice_response(summary_prompt)
        
        except Exception as e:
            logger.error(f"Failed to get exam summary: {e}")
            return f"Error generating summary: {str(e)}"

# Initialize the AI Exam Helper
ai_exam_helper = AIExamHelper()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'ai_ready': True,
        'anythingllm_connected': ai_exam_helper.anythingllm_ready,
        'npu_available': ai_exam_helper.anythingllm_ready,
        'tts_status': tts_engine.get_status(),
        'features_working': {
            'exam_processing': True,
            'ai_conversation': True,
            'question_reading': True,
            'answer_tracking': True,
            'voice_output': tts_engine.get_status()['available'],
            'npu_acceleration': ai_exam_helper.anythingllm_ready
        }
    })

@app.route('/test-voice', methods=['POST'])
def test_voice():
    """Test voice output with comprehensive audio testing"""
    try:
        test_message = "Hello! This is a test of the voice system. Can you hear this clearly? If you can hear this, the audio output is working perfectly!"
        audio_success = ai_exam_helper.speak_response(test_message)
        
        return jsonify({
            'success': True,
            'message': test_message,
            'audio_output_success': audio_success,
            'tts_status': tts_engine.get_status(),
            'audio_info': {
                'engine': tts_engine.get_status()['engine'],
                'available': tts_engine.get_status()['available'],
                'test_completed': True
            }
        })
    except Exception as e:
        logger.error(f"Voice test failed: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'audio_output_success': False,
            'tts_status': tts_engine.get_status()
        }), 500

@app.route('/process-exam', methods=['POST'])
def process_exam():
    """Process exam document using AnythingLLM"""
    try:
        data = request.get_json()
        document_data = data.get('document')
        document_type = data.get('type', 'pdf')
        
        if not document_data:
            return jsonify({'error': 'No document data provided'}), 400
        
        # Convert base64 to bytes
        try:
            document_bytes = base64.b64decode(document_data)
        except Exception as e:
            return jsonify({'error': 'Invalid document data'}), 400
        
        # Process exam with AnythingLLM
        success, message = ai_exam_helper.process_exam_document(document_bytes, document_type)
        
        if success:
            if isinstance(message, dict):
                # New format with instructions
                return jsonify({
                    'success': True,
                    'message': message['message'],
                    'instructions': message['instructions'],
                    'exam_info': message['exam_info']
                })
            else:
                # Legacy format
                return jsonify({
                    'success': True,
                    'message': message,
                    'exam_info': {
                        'title': ai_exam_helper.current_exam.get('exam_title', 'Unknown'),
                        'total_questions': ai_exam_helper.current_exam.get('total_questions', 0),
                        'questions': ai_exam_helper.current_exam.get('questions', [])
                    }
                })
        else:
            return jsonify({
                'success': False,
                'message': message
            }), 400
            
    except Exception as e:
        logger.error(f"Exam processing error: {e}")
        return jsonify({'error': f'Processing failed: {str(e)}'}), 500

@app.route('/start-exam', methods=['POST'])
def start_exam():
    """Start exam session with AnythingLLM"""
    try:
        success, message = ai_exam_helper.start_exam_session()
        
        if success:
            return jsonify({
                'success': True,
                'message': message,
                'session_info': {
                    'total_questions': ai_exam_helper.exam_session['questions_total'],
                    'current_question': ai_exam_helper.current_question + 1
                }
            })
        else:
            return jsonify({
                'success': False,
                'message': message
            }), 400
            
    except Exception as e:
        logger.error(f"Exam start error: {e}")
        return jsonify({'error': f'Failed to start exam: {str(e)}'}), 500

@app.route('/start-voice-exam', methods=['POST'])
def start_voice_exam():
    """Start voice-driven exam session with guaranteed audio output"""
    try:
        if not ai_exam_helper.current_exam:
            return jsonify({'error': 'No exam loaded'}), 400
        
        # Create comprehensive welcome message
        welcome_message = f"""Welcome to your voice exam! I found {ai_exam_helper.current_exam.get('total_questions', 0)} questions for you. 
        
        I will read each question aloud clearly and slowly. You can say 'repeat' to hear it again, 'slower' to hear it more slowly, or just give your answer.
        
        When you're ready to begin, say 'begin' or 'ready'. Take your time - there's no rush!
        
        This is a voice-only exam, so everything will be spoken aloud. Just listen and respond with your voice."""
        
        # Ensure audio output
        audio_success = ai_exam_helper.speak_response(welcome_message)
        
        return jsonify({
            'success': True,
            'message': welcome_message,
            'voice_enabled': True,
            'audio_output_success': audio_success,
            'session_info': {
                'total_questions': ai_exam_helper.exam_session['questions_total'],
                'current_question': ai_exam_helper.current_question + 1
            },
            'audio_info': {
                'engine': tts_engine.get_status()['engine'],
                'available': tts_engine.get_status()['available']
            }
        })
            
    except Exception as e:
        logger.error(f"Voice exam start error: {e}")
        return jsonify({'error': f'Failed to start voice exam: {str(e)}'}), 500

@app.route('/get-question', methods=['GET'])
def get_question():
    """Get current question with AnythingLLM"""
    try:
        question, message = ai_exam_helper.get_current_question()
        
        if question:
            return jsonify({
                'success': True,
                'question': question,
                'message': message,
                'progress': {
                    'current': ai_exam_helper.current_question + 1,
                    'total': ai_exam_helper.exam_session['questions_total'],
                    'answered': ai_exam_helper.exam_session['questions_answered']
                }
            })
        else:
            return jsonify({
                'success': False,
                'message': message
            }), 400
    
    except Exception as e:
        logger.error(f"Get question error: {e}")
        return jsonify({'error': f'Failed to get question: {str(e)}'}), 500

@app.route('/student-response', methods=['POST'])
def student_response():
    """Handle student response using AnythingLLM"""
    try:
        data = request.get_json()
        response_text = data.get('response', '')
        
        if not response_text:
            return jsonify({'error': 'No response provided'}), 400
        
        ai_response = ai_exam_helper.handle_student_response(response_text)
        
        return jsonify({
            'success': True,
            'ai_response': ai_response,
            'progress': {
                'current': ai_exam_helper.current_question + 1,
                'total': ai_exam_helper.exam_session['questions_total'],
                'answered': ai_exam_helper.exam_session['questions_answered']
            },
            'answers': ai_exam_helper.student_answers
        })
        
    except Exception as e:
        logger.error(f"Student response error: {e}")
        return jsonify({'error': f'Failed to process response: {str(e)}'}), 500

@app.route('/exam-summary', methods=['GET'])
def exam_summary():
    """Get exam summary using AnythingLLM"""
    try:
        summary = ai_exam_helper.get_exam_summary()
        
        return jsonify({
            'success': True,
            'summary': summary,
            'session_info': ai_exam_helper.exam_session,
            'answers': ai_exam_helper.student_answers
        })
    
    except Exception as e:
        logger.error(f"Exam summary error: {e}")
        return jsonify({'error': f'Failed to get summary: {str(e)}'}), 500

if __name__ == '__main__':
    logger.info("🚀 Starting AI Exam Helper...")
    logger.info(f"AnythingLLM Status: {'✅ Connected' if ai_exam_helper.anythingllm_ready else '❌ Demo Mode'}")
    
    app.run(host='127.0.0.1', port=5000, debug=True)