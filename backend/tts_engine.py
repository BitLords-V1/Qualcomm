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
        """Speak text using system TTS"""
        if not self.tts_engine or not text.strip():
            logger.warning("TTS not available or no text to speak")
            return
        
        try:
            # Stop any current speech
            self.stop_speaking()
            
            # Update current text
            self.current_text = text
            logger.info(f"Speaking text: {text[:50]}...")
            
            # Speak directly (not in thread for now to debug)
            if self.tts_engine == 'say':
                # macOS say command - try Alex voice first, fallback to default
                try:
                    subprocess.run(['say', '-r', '150', '-v', 'Alex', text], check=True)
                    logger.info("TTS completed successfully")
                except Exception as e:
                    logger.error(f"Alex voice failed: {e}, trying default")
                    subprocess.run(['say', '-r', '150', text], check=True)
                    logger.info("TTS completed with default voice")
            elif self.tts_engine == 'espeak':
                # Linux espeak command
                subprocess.run(['espeak', '-s', '200', '-v', 'en+f3', text], check=True)
            elif self.tts_engine == 'sapi':
                # Windows SAPI
                escaped_text = text.replace('"', '\\"')
                subprocess.run([
                    'powershell', '-Command',
                    f'Add-Type -AssemblyName System.Speech; (New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak("{escaped_text}")'
                ], check=True)
            
            self.is_speaking = False
            
        except Exception as e:
            logger.error(f"Failed to speak text: {e}")
            self.is_speaking = False
    
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
