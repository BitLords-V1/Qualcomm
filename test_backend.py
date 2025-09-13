"""
Test script for iLumina Backend
Tests all endpoints and functionality
"""

import requests
import base64
import json
import time
import cv2
import numpy as np
from PIL import Image
import io

class ILuminaBackendTester:
    def __init__(self, base_url="http://127.0.0.1:5000"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def test_health(self):
        """Test health endpoint"""
        print("Testing health endpoint...")
        try:
            response = self.session.get(f"{self.base_url}/health")
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Health check passed: {data}")
                return True
            else:
                print(f"❌ Health check failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Health check error: {e}")
            return False
    
    def create_test_image(self):
        """Create a test image with text"""
        # Create a simple test image
        img = np.ones((400, 600, 3), dtype=np.uint8) * 255
        
        # Add some text
        cv2.putText(img, "Hello World", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
        cv2.putText(img, "This is a test", (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        # Convert to base64
        _, buffer = cv2.imencode('.jpg', img)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return f"data:image/jpeg;base64,{img_base64}"
    
    def test_ocr(self):
        """Test OCR endpoint"""
        print("Testing OCR endpoint...")
        try:
            test_image = self.create_test_image()
            
            response = self.session.post(f"{self.base_url}/ocr", json={
                "image": test_image
            })
            
            if response.status_code == 200:
                data = response.json()
                if data.get('success'):
                    print(f"✅ OCR test passed: {data['text']}")
                    return True
                else:
                    print(f"❌ OCR test failed: {data.get('error', 'Unknown error')}")
                    return False
            else:
                print(f"❌ OCR test failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ OCR test error: {e}")
            return False
    
    def create_test_audio(self):
        """Create a test audio file"""
        # Create a simple sine wave audio
        sample_rate = 16000
        duration = 2  # seconds
        frequency = 440  # Hz
        
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        audio = np.sin(2 * np.pi * frequency * t)
        
        # Convert to 16-bit PCM
        audio_16bit = (audio * 32767).astype(np.int16)
        
        # Create WAV file in memory
        import wave
        audio_io = io.BytesIO()
        
        with wave.open(audio_io, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 2 bytes per sample
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_16bit.tobytes())
        
        audio_io.seek(0)
        audio_base64 = base64.b64encode(audio_io.read()).decode('utf-8')
        
        return f"data:audio/wav;base64,{audio_base64}"
    
    def test_stt(self):
        """Test speech-to-text endpoint"""
        print("Testing STT endpoint...")
        try:
            test_audio = self.create_test_audio()
            
            response = self.session.post(f"{self.base_url}/stt", json={
                "audio": test_audio
            })
            
            if response.status_code == 200:
                data = response.json()
                if data.get('success'):
                    print(f"✅ STT test passed: {data['text']}")
                    return True
                else:
                    print(f"❌ STT test failed: {data.get('error', 'Unknown error')}")
                    return False
            else:
                print(f"❌ STT test failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ STT test error: {e}")
            return False
    
    def test_command(self):
        """Test command endpoint"""
        print("Testing command endpoint...")
        try:
            test_commands = [
                "repeat",
                "slower",
                "faster",
                "spell",
                "hello world"
            ]
            
            for command in test_commands:
                response = self.session.post(f"{self.base_url}/command", json={
                    "text": command
                })
                
                if response.status_code == 200:
                    data = response.json()
                    if data.get('success'):
                        print(f"✅ Command '{command}' passed: {data['action']}")
                    else:
                        print(f"❌ Command '{command}' failed: {data.get('error', 'Unknown error')}")
                        return False
                else:
                    print(f"❌ Command '{command}' failed: {response.status_code}")
                    return False
            
            return True
        except Exception as e:
            print(f"❌ Command test error: {e}")
            return False
    
    def test_tts(self):
        """Test text-to-speech endpoint"""
        print("Testing TTS endpoint...")
        try:
            response = self.session.post(f"{self.base_url}/tts", json={
                "text": "Hello, this is a test of the text to speech system."
            })
            
            if response.status_code == 200:
                data = response.json()
                if data.get('success'):
                    print("✅ TTS test passed")
                    return True
                else:
                    print(f"❌ TTS test failed: {data.get('error', 'Unknown error')}")
                    return False
            else:
                print(f"❌ TTS test failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ TTS test error: {e}")
            return False
    
    def run_all_tests(self):
        """Run all tests"""
        print("iLumina Backend Test Suite")
        print("=" * 30)
        
        tests = [
            ("Health Check", self.test_health),
            ("OCR", self.test_ocr),
            ("STT", self.test_stt),
            ("Command", self.test_command),
            ("TTS", self.test_tts)
        ]
        
        passed = 0
        total = len(tests)
        
        for test_name, test_func in tests:
            print(f"\n--- {test_name} ---")
            if test_func():
                passed += 1
            else:
                print(f"❌ {test_name} failed")
        
        print(f"\n{'='*30}")
        print(f"Tests passed: {passed}/{total}")
        
        if passed == total:
            print("🎉 All tests passed!")
            return True
        else:
            print("❌ Some tests failed")
            return False

def main():
    """Main test function"""
    tester = ILuminaBackendTester()
    
    # Wait for backend to start
    print("Waiting for backend to start...")
    time.sleep(5)
    
    success = tester.run_all_tests()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
