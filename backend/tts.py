import pyttsx3

class TTS:
    def __init__(self, rate=180, volume=1.0, voice=None):
        self.eng = pyttsx3.init()  # SAPI5 on Windows, fully offline
        self.rate = rate
        self.volume = volume
        self.voice = voice
        self._apply()

    def _apply(self):
        self.eng.setProperty('rate', self.rate)
        self.eng.setProperty('volume', self.volume)
        if self.voice:
            self.eng.setProperty('voice', self.voice)

    def say(self, text: str):
        if not text:
            return
        self.eng.stop()
        self.eng.say(text)
        self.eng.runAndWait()

    def slower(self, step=20):
        self.rate = max(80, self.rate - step)
        self._apply()

    def faster(self, step=20):
        self.rate = min(320, self.rate + step)
        self._apply()
