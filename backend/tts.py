# tts_macos.py
import threading
import platform
import logging
import pyttsx3

logger = logging.getLogger(__name__)

class TTS:
    def __init__(self, rate: int = 180, volume: float = 0.9, voice_preference=None):
        """
        macOS offline TTS using NSSpeechSynthesizer via pyttsx3.
        voice_preference: optional list of preferred voice names (lowercase substrings).
        """
        self._lock = threading.Lock()
        self._engine = None
        self._is_speaking = False
        self._rate = max(50, min(400, int(rate)))
        self._volume = max(0.0, min(1.0, float(volume)))
        self._voice_preference = voice_preference or ["samantha", "ava", "victoria", "alex", "moira", "daniel"]
        self.available = False

        try:
            if platform.system() == "Darwin":
                self._engine = pyttsx3.init(driverName="nsss")
            else:
                self._engine = pyttsx3.init()

            with self._lock:
                # Set voice
                chosen = None
                voices = self._engine.getProperty("voices") or []
                for name in self._voice_preference:
                    for v in voices:
                        if name in (v.name or "").lower():
                            chosen = v.id
                            break
                    if chosen:
                        break
                if not chosen and voices:
                    chosen = voices[0].id
                if chosen:
                    self._engine.setProperty("voice", chosen)

                self._engine.setProperty("rate", self._rate)
                self._engine.setProperty("volume", self._volume)

            self.available = True
            logger.info("TTS initialized (macOS NSSpeechSynthesizer).")
        except Exception as e:
            logger.error(f"TTS init failed: {e}")
            self._engine = None
            self.available = False

    @property
    def rate(self) -> int:
        return self._rate

    def slower(self, step: int = 30):
        self._rate = max(50, self._rate - int(step))
        if self._engine:
            with self._lock:
                self._engine.setProperty("rate", self._rate)

    def faster(self, step: int = 30):
        self._rate = min(400, self._rate + int(step))
        if self._engine:
            with self._lock:
                self._engine.setProperty("rate", self._rate)

    def say(self, text: str):
        if not self._engine or not (text or "").strip():
            return
        def _worker():
            try:
                self._is_speaking = True
                with self._lock:
                    self._engine.say(text)
                    self._engine.runAndWait()
            except Exception as e:
                logger.error(f"TTS error: {e}")
            finally:
                self._is_speaking = False
        t = threading.Thread(target=_worker, daemon=True)
        t.start()