"""
TTS Engine for iLumina
Simple Text-to-Speech using system commands
"""

import os
import logging
import threading
import subprocess

logger = logging.getLogger(__name__)

class TTSEngine:
    def __init__(self):
        """Initialize TTS engine"""
        self.tts_engine = None
        self.current_text = ""
        self.is_speaking = False
        self._initialize_tts()
    
    def _initialize_tts(self):
        """Initialize TTS engine using system commands"""
        try:
            # Try macOS say command first
            result = subprocess.run(['which', 'say'], capture_output=True, text=True)
            if result.returncode == 0:
                self.tts_engine = 'say'
                logger.info("✅ TTS engine initialized with macOS 'say' command!")
                return True
            
            # Try espeak (Linux)
            result = subprocess.run(['which', 'espeak'], capture_output=True, text=True)
            if result.returncode == 0:
                self.tts_engine = 'espeak'
                logger.info("✅ TTS engine initialized with espeak!")
                return True
            
            # Try Windows SAPI
            if os.name == 'nt':
                self.tts_engine = 'sapi'
                logger.info("✅ TTS engine initialized with Windows SAPI!")
                return True
            
            logger.warning("⚠️ No TTS system command available")
            return False
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize TTS: {e}")
            return False
    
    def speak(self, text):
        """Speak text using system TTS with enhanced audio output"""
        if not self.tts_engine or not text.strip():
            logger.warning("TTS not available or no text to speak")
            return False
        
        try:
            # Stop any current speech
            self.stop_speaking()
            
            # Update current text
            self.current_text = text
            logger.info(f"🔊 Speaking text: {text[:50]}...")
            
            # Speak with enhanced audio output
            if self.tts_engine == 'say':
                # macOS say command with better voice and rate
                try:
                    # Try with Alex voice first (clearer for educational content)
                    result = subprocess.run([
                        'say', 
                        '-r', '160',  # Slightly faster rate for better comprehension
                        '-v', 'Alex',  # Clear, educational voice
                        text
                    ], check=True, capture_output=True, text=True)
                    logger.info("✅ TTS completed successfully with Alex voice")
                    return True
                except Exception as e:
                    logger.warning(f"Alex voice failed: {e}, trying default voice")
                    try:
                        # Fallback to default voice
                        result = subprocess.run([
                            'say', 
                            '-r', '160',
                            text
                        ], check=True, capture_output=True, text=True)
                        logger.info("✅ TTS completed successfully with default voice")
                        return True
                    except Exception as e2:
                        logger.error(f"Default voice also failed: {e2}")
                        return False
                        
            elif self.tts_engine == 'espeak':
                # Linux espeak command
                result = subprocess.run([
                    'espeak', 
                    '-s', '180',  # Speed
                    '-v', 'en+f3',  # Female voice
                    '-a', '200',  # Amplitude
                    text
                ], check=True, capture_output=True, text=True)
                logger.info("✅ TTS completed successfully with espeak")
                return True
                
            elif self.tts_engine == 'sapi':
                # Windows SAPI with PowerShell
                escaped_text = text.replace('"', '\\"').replace("'", "\\'")
                cmd = f'Add-Type -AssemblyName System.Speech; $synth = New-Object System.Speech.Synthesis.SpeechSynthesizer; $synth.Rate = 2; $synth.Speak("{escaped_text}")'
                result = subprocess.run([
                    'powershell', '-Command', cmd
                ], check=True, capture_output=True, text=True)
                logger.info("✅ TTS completed successfully with Windows SAPI")
                return True
            
            self.is_speaking = False
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to speak text: {e}")
            self.is_speaking = False
            return False
    
    def stop_speaking(self):
        """Stop current speech"""
        if self.is_speaking:
            try:
                if self.tts_engine == 'say':
                    subprocess.run(['pkill', '-f', 'say'], capture_output=True)
                elif self.tts_engine == 'espeak':
                    subprocess.run(['pkill', '-f', 'espeak'], capture_output=True)
                elif self.tts_engine == 'sapi':
                    subprocess.run(['taskkill', '/f', '/im', 'powershell.exe'], capture_output=True)
                self.is_speaking = False
            except Exception as e:
                logger.error(f"Failed to stop speech: {e}")
    
    def repeat(self):
        """Repeat the last spoken text"""
        if self.current_text.strip():
            self.speak(self.current_text)
    
    def get_status(self):
        """Get TTS engine status"""
        return {
            'available': self.tts_engine is not None,
            'engine': self.tts_engine,
            'is_speaking': self.is_speaking,
            'current_text': self.current_text
        }

# Global TTS instance
tts_engine = TTSEngine()
