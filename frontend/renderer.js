/**
 * iLumina Renderer Process - Frontend Application Logic
 * Handles UI interactions, camera, audio, and backend communication
 */

const { ipcRenderer } = require('electron');
const axios = require('axios');

class ILuminaRenderer {
    constructor() {
        this.backendUrl = 'http://127.0.0.1:5000';
        this.isRecording = false;
        this.mediaRecorder = null;
        this.audioChunks = [];
        this.currentText = '';
        this.isBackendReady = false;
        
        // Initialize the application
        this.init();
    }

    async init() {
        console.log('Initializing iLumina renderer...');
        
        // Setup event listeners
        this.setupEventListeners();
        
        // Initialize camera
        await this.initializeCamera();
        
        // Check backend status
        await this.checkBackendStatus();
        
        // Update UI
        this.updateUI();
        
        console.log('iLumina renderer initialized successfully');
    }

    setupEventListeners() {
        // Camera controls
        document.getElementById('capture-btn').addEventListener('click', () => {
            this.captureAndOCR();
        });

        // Voice controls
        document.getElementById('start-recording').addEventListener('click', () => {
            this.startVoiceRecording();
        });

        document.getElementById('stop-recording').addEventListener('click', () => {
            this.stopVoiceRecording();
        });

        // Text action buttons
        document.getElementById('repeat-btn').addEventListener('click', () => {
            this.executeCommand('repeat');
        });

        document.getElementById('slower-btn').addEventListener('click', () => {
            this.executeCommand('slower');
        });

        document.getElementById('faster-btn').addEventListener('click', () => {
            this.executeCommand('faster');
        });

        document.getElementById('spell-btn').addEventListener('click', () => {
            this.executeCommand('spell');
        });

        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (e.ctrlKey || e.metaKey) {
                switch (e.key) {
                    case 'r':
                        e.preventDefault();
                        this.captureAndOCR();
                        break;
                    case ' ':
                        e.preventDefault();
                        if (this.isRecording) {
                            this.stopVoiceRecording();
                        } else {
                            this.startVoiceRecording();
                        }
                        break;
                }
            }
        });

        // Check backend status periodically
        setInterval(() => {
            this.checkBackendStatus();
        }, 5000);
    }

    async initializeCamera() {
        try {
            const video = document.getElementById('webcam');
            const stream = await navigator.mediaDevices.getUserMedia({
                video: {
                    width: { ideal: 1280 },
                    height: { ideal: 720 },
                    facingMode: 'environment'
                }
            });
            
            video.srcObject = stream;
            video.play();
            
            console.log('Camera initialized successfully');
            this.showToast('Camera ready', 'success');
            
        } catch (error) {
            console.error('Camera initialization failed:', error);
            this.showToast('Camera access denied. Please enable camera permissions.', 'error');
        }
    }

    async checkBackendStatus() {
        try {
            const response = await axios.get(`${this.backendUrl}/health`, { timeout: 2000 });
            
            if (response.status === 200) {
                this.isBackendReady = true;
                this.updateBackendStatus(true);
                this.updateNPUStatus(response.data.npu_available);
            } else {
                this.isBackendReady = false;
                this.updateBackendStatus(false);
            }
            
        } catch (error) {
            this.isBackendReady = false;
            this.updateBackendStatus(false);
        }
    }

    updateBackendStatus(isReady) {
        const backendDot = document.getElementById('backend-dot');
        const backendText = document.getElementById('backend-text');
        
        if (isReady) {
            backendDot.className = 'status-dot online';
            backendText.textContent = 'Backend Ready';
        } else {
            backendDot.className = 'status-dot offline';
            backendText.textContent = 'Backend Offline';
        }
    }

    updateNPUStatus(npuAvailable) {
        const npuDot = document.getElementById('npu-dot');
        const npuText = document.getElementById('npu-text');
        
        if (npuAvailable) {
            npuDot.className = 'status-dot npu';
            npuText.textContent = 'NPU Active';
        } else {
            npuDot.className = 'status-dot offline';
            npuText.textContent = 'CPU Fallback';
        }
    }

    async captureAndOCR() {
        if (!this.isBackendReady) {
            this.showToast('Backend not ready. Please wait...', 'error');
            return;
        }

        try {
            this.showLoading('Processing image with OCR...');
            
            // Capture image from webcam
            const video = document.getElementById('webcam');
            const canvas = document.createElement('canvas');
            const ctx = canvas.getContext('2d');
            
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            ctx.drawImage(video, 0, 0);
            
            // Convert to base64
            const imageData = canvas.toDataURL('image/jpeg', 0.8);
            
            // Send to backend
            const response = await axios.post(`${this.backendUrl}/ocr`, {
                image: imageData
            });
            
            if (response.data.success) {
                this.currentText = response.data.text;
                this.displayText(response.data);
                this.speakText(response.data.text);
                this.showToast('Text extracted successfully!', 'success');
            } else {
                throw new Error(response.data.error || 'OCR failed');
            }
            
        } catch (error) {
            console.error('OCR error:', error);
            this.showToast(`OCR failed: ${error.message}`, 'error');
        } finally {
            this.hideLoading();
        }
    }

    async startVoiceRecording() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            
            this.mediaRecorder = new MediaRecorder(stream, {
                mimeType: 'audio/webm;codecs=opus'
            });
            
            this.audioChunks = [];
            
            this.mediaRecorder.ondataavailable = (event) => {
                this.audioChunks.push(event.data);
            };
            
            this.mediaRecorder.onstop = () => {
                this.processAudioRecording();
            };
            
            this.mediaRecorder.start();
            this.isRecording = true;
            
            // Update UI
            document.getElementById('start-recording').disabled = true;
            document.getElementById('stop-recording').disabled = false;
            
            this.showToast('Recording started. Speak now...', 'info');
            
        } catch (error) {
            console.error('Voice recording failed:', error);
            this.showToast('Microphone access denied. Please enable microphone permissions.', 'error');
        }
    }

    stopVoiceRecording() {
        if (this.mediaRecorder && this.isRecording) {
            this.mediaRecorder.stop();
            this.isRecording = false;
            
            // Update UI
            document.getElementById('start-recording').disabled = false;
            document.getElementById('stop-recording').disabled = true;
            
            this.showToast('Processing voice command...', 'info');
        }
    }

    async processAudioRecording() {
        if (!this.isBackendReady) {
            this.showToast('Backend not ready. Please wait...', 'error');
            return;
        }

        try {
            this.showLoading('Processing speech...');
            
            // Convert audio chunks to blob
            const audioBlob = new Blob(this.audioChunks, { type: 'audio/webm' });
            
            // Convert to base64
            const reader = new FileReader();
            reader.onload = async () => {
                try {
                    const audioData = reader.result;
                    
                    // Send to backend for speech-to-text
                    const response = await axios.post(`${this.backendUrl}/stt`, {
                        audio: audioData
                    });
                    
                    if (response.data.success) {
                        const command = response.data.text;
                        this.showToast(`Heard: "${command}"`, 'info');
                        await this.executeCommand(command);
                    } else {
                        throw new Error(response.data.error || 'Speech recognition failed');
                    }
                    
                } catch (error) {
                    console.error('Speech processing error:', error);
                    this.showToast(`Speech processing failed: ${error.message}`, 'error');
                } finally {
                    this.hideLoading();
                }
            };
            
            reader.readAsDataURL(audioBlob);
            
        } catch (error) {
            console.error('Audio processing error:', error);
            this.showToast(`Audio processing failed: ${error.message}`, 'error');
            this.hideLoading();
        }
    }

    async executeCommand(command) {
        if (!this.isBackendReady) {
            this.showToast('Backend not ready. Please wait...', 'error');
            return;
        }

        try {
            this.showLoading('Processing command...');
            
            const response = await axios.post(`${this.backendUrl}/command`, {
                text: command
            });
            
            if (response.data.success) {
                const result = response.data;
                
                // Update UI based on action
                if (result.action === 'repeat' && result.tts_text) {
                    this.speakText(result.tts_text);
                } else if (result.action === 'spell' && result.tts_text) {
                    this.speakText(result.tts_text);
                } else if (result.action === 'slower' || result.action === 'faster') {
                    this.showToast(result.response, 'info');
                    this.speakText(result.tts_text);
                } else if (result.action === 'stop') {
                    this.showToast('Speech stopped', 'info');
                } else {
                    this.showToast(result.response, 'info');
                    if (result.tts_text) {
                        this.speakText(result.tts_text);
                    }
                }
            } else {
                throw new Error(response.data.error || 'Command processing failed');
            }
            
        } catch (error) {
            console.error('Command execution error:', error);
            this.showToast(`Command failed: ${error.message}`, 'error');
        } finally {
            this.hideLoading();
        }
    }

    async speakText(text) {
        if (!this.isBackendReady || !text) {
            return;
        }

        try {
            await axios.post(`${this.backendUrl}/tts`, {
                text: text
            });
        } catch (error) {
            console.error('TTS error:', error);
            this.showToast('Text-to-speech failed', 'error');
        }
    }

    displayText(ocrResult) {
        const textDisplay = document.getElementById('text-display');
        const metadata = document.getElementById('text-metadata');
        
        // Display extracted text
        textDisplay.innerHTML = `<p>${ocrResult.text}</p>`;
        textDisplay.classList.remove('placeholder');
        
        // Display metadata
        const inferenceTime = ocrResult.inference_time ? `${ocrResult.inference_time.toFixed(1)}ms` : 'N/A';
        const provider = ocrResult.npu_used ? 'NPU' : 'CPU';
        const confidence = ocrResult.confidence ? `${(ocrResult.confidence * 100).toFixed(1)}%` : 'N/A';
        
        metadata.innerHTML = `
            <div>Provider: ${provider} | Time: ${inferenceTime} | Confidence: ${confidence}</div>
        `;
    }

    updateUI() {
        // Update offline status
        const offlineStatus = document.getElementById('offline-status');
        const isOnline = navigator.onLine;
        
        if (!isOnline) {
            offlineStatus.querySelector('.status-dot').className = 'status-dot offline';
            offlineStatus.querySelector('span').textContent = 'Offline Mode';
        } else {
            offlineStatus.querySelector('.status-dot').className = 'status-dot online';
            offlineStatus.querySelector('span').textContent = 'Online Mode';
        }
    }

    showLoading(message = 'Processing...') {
        const overlay = document.getElementById('loading-overlay');
        const text = document.getElementById('loading-text');
        
        text.textContent = message;
        overlay.classList.add('show');
    }

    hideLoading() {
        const overlay = document.getElementById('loading-overlay');
        overlay.classList.remove('show');
    }

    showToast(message, type = 'info') {
        const toast = document.getElementById('toast');
        const toastMessage = document.getElementById('toast-message');
        
        toastMessage.textContent = message;
        
        // Update icon based on type
        const icon = toast.querySelector('.toast-icon');
        switch (type) {
            case 'success':
                icon.textContent = '✅';
                break;
            case 'error':
                icon.textContent = '❌';
                break;
            case 'warning':
                icon.textContent = '⚠️';
                break;
            default:
                icon.textContent = 'ℹ️';
        }
        
        toast.classList.add('show');
        
        // Auto-hide after 3 seconds
        setTimeout(() => {
            toast.classList.remove('show');
        }, 3000);
    }
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new ILuminaRenderer();
});
