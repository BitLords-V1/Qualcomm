"""
Local Command Agent for iLumina
Handles voice/text commands and controls offline TTS
"""

import logging
import re
import pyttsx3
import threading
import queue

logger = logging.getLogger(__name__)

class CommandAgent:
    def __init__(self):
        """Initialize command agent with offline TTS"""
        self.tts_engine = None
        self.command_queue = queue.Queue()
        self.is_speaking = False
        self.current_text = ""
        self.speech_rate = 200  # Default speech rate (words per minute)
        
        # Initialize TTS engine
        self._initialize_tts()
        
        # Command patterns for rule-based agent
        self.command_patterns = {
            'repeat': [
                r'\b(repeat|again|say again)\b',
                r'\b(read (it )?again)\b'
            ],
            'slower': [
                r'\b(slower|slow down|speak slower)\b',
                r'\b(read slower)\b'
            ],
            'faster': [
                r'\b(faster|speed up|speak faster)\b',
                r'\b(read faster)\b'
            ],
            'spell': [
                r'\b(spell|spelling|how do you spell)\b',
                r'\b(letter by letter)\b'
            ],
            'stop': [
                r'\b(stop|stop speaking|shut up)\b',
                r'\b(be quiet)\b'
            ]
        }
    
    def _initialize_tts(self):
        """Initialize offline TTS engine"""
        try:
            self.tts_engine = pyttsx3.init()
            
            # Configure TTS properties
            voices = self.tts_engine.getProperty('voices')
            if voices:
                # Try to find a female voice (usually more pleasant)
                for voice in voices:
                    if 'female' in voice.name.lower() or 'zira' in voice.name.lower():
                        self.tts_engine.setProperty('voice', voice.id)
                        break
                else:
                    # Use first available voice
                    self.tts_engine.setProperty('voice', voices[0].id)
            
            # Set speech rate
            self.tts_engine.setProperty('rate', self.speech_rate)
            
            # Set volume (0.0 to 1.0)
            self.tts_engine.setProperty('volume', 0.9)
            
            logger.info("TTS engine initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize TTS engine: {e}")
            self.tts_engine = None
    
    def _match_command(self, text):
        """
        Match text against command patterns
        
        Args:
            text: Input text to analyze
            
        Returns:
            Tuple of (command_type, confidence)
        """
        text_lower = text.lower().strip()
        
        for command_type, patterns in self.command_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    return command_type, 1.0  # High confidence for exact matches
        
        return 'unknown', 0.0
    
    def process_command(self, text):
        """
        Process a text command and return appropriate action
        
        Args:
            text: Command text to process
            
        Returns:
            Dictionary with action, response, and TTS text
        """
        try:
            command_type, confidence = self._match_command(text)
            
            if command_type == 'unknown':
                return {
                    'action': 'unknown',
                    'response': "I didn't understand that command. Try saying 'repeat', 'slower', 'faster', or 'spell'.",
                    'tts_text': "I didn't understand that command. Try saying repeat, slower, faster, or spell."
                }
            
            # Process known commands
            if command_type == 'repeat':
                if self.current_text:
                    return {
                        'action': 'repeat',
                        'response': f"Repeating: {self.current_text}",
                        'tts_text': self.current_text
                    }
                else:
                    return {
                        'action': 'repeat',
                        'response': "Nothing to repeat. Please capture some text first.",
                        'tts_text': "Nothing to repeat. Please capture some text first."
                    }
            
            elif command_type == 'slower':
                self.speech_rate = max(50, self.speech_rate - 50)  # Decrease by 50 WPM
                if self.tts_engine:
                    self.tts_engine.setProperty('rate', self.speech_rate)
                return {
                    'action': 'slower',
                    'response': f"Speech rate set to {self.speech_rate} words per minute",
                    'tts_text': f"Speaking slower at {self.speech_rate} words per minute"
                }
            
            elif command_type == 'faster':
                self.speech_rate = min(400, self.speech_rate + 50)  # Increase by 50 WPM
                if self.tts_engine:
                    self.tts_engine.setProperty('rate', self.speech_rate)
                return {
                    'action': 'faster',
                    'response': f"Speech rate set to {self.speech_rate} words per minute",
                    'tts_text': f"Speaking faster at {self.speech_rate} words per minute"
                }
            
            elif command_type == 'spell':
                if self.current_text:
                    spelled_text = ' '.join([char for char in self.current_text if char.isalnum() or char.isspace()])
                    return {
                        'action': 'spell',
                        'response': f"Spelling: {spelled_text}",
                        'tts_text': spelled_text
                    }
                else:
                    return {
                        'action': 'spell',
                        'response': "Nothing to spell. Please capture some text first.",
                        'tts_text': "Nothing to spell. Please capture some text first."
                    }
            
            elif command_type == 'stop':
                self.stop_speaking()
                return {
                    'action': 'stop',
                    'response': "Stopped speaking",
                    'tts_text': ""
                }
            
            return {
                'action': 'unknown',
                'response': "Command processed but no action taken",
                'tts_text': ""
            }
            
        except Exception as e:
            logger.error(f"Command processing error: {e}")
            return {
                'action': 'error',
                'response': f"Error processing command: {str(e)}",
                'tts_text': "Error processing command"
            }
    
    def speak(self, text):
        """
        Speak text using offline TTS
        
        Args:
            text: Text to speak
        """
        if not self.tts_engine or not text.strip():
            return
        
        try:
            # Stop any current speech
            self.stop_speaking()
            
            # Update current text for repeat functionality
            self.current_text = text
            
            # Speak in a separate thread to avoid blocking
            def speak_thread():
                try:
                    self.is_speaking = True
                    self.tts_engine.say(text)
                    self.tts_engine.runAndWait()
                except Exception as e:
                    logger.error(f"TTS error: {e}")
                finally:
                    self.is_speaking = False
            
            thread = threading.Thread(target=speak_thread)
            thread.daemon = True
            thread.start()
            
        except Exception as e:
            logger.error(f"Failed to speak text: {e}")
    
    def stop_speaking(self):
        """Stop current speech"""
        if self.tts_engine and self.is_speaking:
            try:
                self.tts_engine.stop()
                self.is_speaking = False
            except Exception as e:
                logger.error(f"Failed to stop speech: {e}")
    
    def set_text(self, text):
        """Set current text for repeat/spell functionality"""
        self.current_text = text
    
    def get_status(self):
        """Get current agent status"""
        return {
            'is_speaking': self.is_speaking,
            'speech_rate': self.speech_rate,
            'current_text': self.current_text,
            'tts_available': self.tts_engine is not None
        }
