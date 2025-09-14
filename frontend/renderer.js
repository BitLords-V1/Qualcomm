/**
 * iLumina Renderer Process - Frontend Application Logic
 * Handles UI interactions, camera, audio, and backend communication
 * Browser-compatible version
 */

class ILuminaRenderer {
    constructor() {
        this.backendUrl = 'http://127.0.0.1:5005';
        this.isRecording = false;
        this.mediaRecorder = null;
        this.audioChunks = [];
        this.currentText = '';
        this.isBackendReady = false;
        
        // Initialize the application when DOM is ready
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', () => this.init());
        } else {
            // Wait a bit for axios to load if DOM is already ready
            setTimeout(() => this.init(), 100);
        }
    }

    async init() {
        console.log('Initializing iLumina renderer...');
        console.log('Backend URL:', this.backendUrl);
        
        // Check if axios is loaded, use fetch as fallback
        if (typeof axios === 'undefined') {
            console.warn('Axios not loaded! Using fetch as fallback...');
            this.useFetch = true;
        } else {
            this.useFetch = false;
        }
        
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

    async makeRequest(url, options = {}) {
        if (this.useFetch) {
            // Use fetch as fallback
            const response = await fetch(url, {
                method: options.method || 'GET',
                headers: {
                    'Content-Type': 'application/json',
                    ...options.headers
                },
                body: options.data ? JSON.stringify(options.data) : undefined,
                ...options
            });
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            const data = await response.json();
            return { data, status: response.status };
        } else {
            // Use axios
            return await axios(url, options);
        }
    }

    setupEventListeners() {
        // Test backend button
        document.getElementById('test-backend').addEventListener('click', () => {
            this.testBackendConnection();
        });

        // Test display button
        document.getElementById('test-display').addEventListener('click', () => {
            this.testTextDisplay();
        });

        // Camera controls
        document.getElementById('capture-btn').addEventListener('click', () => {
            this.captureAndOCR();
        });

        // File upload controls
        document.getElementById('upload-btn').addEventListener('click', () => {
            document.getElementById('file-input').click();
        });

        document.getElementById('file-input').addEventListener('change', (e) => {
            this.handleFileUpload(e);
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
            console.log('Initializing camera...');
            const video = document.getElementById('webcam');
            if (!video) {
                throw new Error('Video element not found');
            }

            // Check if getUserMedia is supported
            if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
                throw new Error('Camera not supported in this browser');
            }

            console.log('Requesting camera access...');
            let stream;
            try {
                // Try environment camera first (back camera on mobile)
                stream = await navigator.mediaDevices.getUserMedia({
                    video: {
                        width: { ideal: 1280 },
                        height: { ideal: 720 },
                        facingMode: { ideal: 'environment' }
                    }
                });
            } catch (envError) {
                console.log('Environment camera failed, trying user camera...');
                // Fallback to user camera (front camera)
                stream = await navigator.mediaDevices.getUserMedia({
                    video: {
                        width: { ideal: 1280 },
                        height: { ideal: 720 },
                        facingMode: { ideal: 'user' }
                    }
                });
            }
            
            video.srcObject = stream;
            video.play();
            
            console.log('Camera initialized successfully');
            this.showToast('Camera ready', 'success');
            
        } catch (error) {
            console.error('Camera initialization failed:', error);
            let errorMessage = 'Camera access failed. ';
            
            if (error.name === 'NotAllowedError') {
                errorMessage += 'Please allow camera access and refresh the page.';
            } else if (error.name === 'NotFoundError') {
                errorMessage += 'No camera found on this device.';
            } else if (error.name === 'NotSupportedError') {
                errorMessage += 'Camera not supported. Try using HTTPS or a different browser.';
            } else {
                errorMessage += error.message;
            }
            
            this.showToast(errorMessage, 'error');
            
            // Show a fallback message
            const video = document.getElementById('webcam');
            if (video) {
                video.style.display = 'none';
                const fallback = document.createElement('div');
                fallback.innerHTML = `
                    <div style="padding: 20px; text-align: center; background: #f0f0f0; border-radius: 8px;">
                        <h3>Camera Not Available</h3>
                        <p>Please allow camera access or try:</p>
                        <ul style="text-align: left; display: inline-block;">
                            <li>Refresh the page and allow camera access</li>
                            <li>Use HTTPS (https://127.0.0.1:5005)</li>
                            <li>Try a different browser</li>
                        </ul>
                        <button onclick="location.reload()" style="margin-top: 10px; padding: 8px 16px; background: #007bff; color: white; border: none; border-radius: 4px; cursor: pointer;">
                            Refresh Page
                        </button>
                    </div>
                `;
                video.parentNode.appendChild(fallback);
            }
        }
    }

    async testBackendConnection() {
        console.log('Testing backend connection...');
        
        try {
            const response = await this.makeRequest(`${this.backendUrl}/api/healthz`, { 
                method: 'GET',
                timeout: 5000 
            });
            console.log('Test successful! Backend response:', response.data);
            this.showToast('Backend connection successful!', 'success');
            this.isBackendReady = true;
            this.updateBackendStatus(true);
        } catch (error) {
            console.error('Test failed:', error);
            this.showToast(`Backend test failed: ${error.message}`, 'error');
        }
    }

    testTextDisplay() {
        console.log('Testing text display...');
        
        // Create a test OCR result
        const testResult = {
            text: "This is a test text to verify that the display function is working correctly. The text should appear in the Extracted Text section.",
            confidence: 0.95,
            success: true
        };
        
        console.log('Test OCR result:', testResult);
        this.displayText(testResult);
        this.showToast('Test text displayed!', 'success');
    }

    async checkBackendStatus() {
        try {
            console.log('Checking backend status at:', `${this.backendUrl}/api/healthz`);
            const response = await this.makeRequest(`${this.backendUrl}/api/healthz`, { 
                method: 'GET',
                timeout: 2000 
            });
            
            console.log('Backend response:', response.data);
            if (response.status === 200) {
                this.isBackendReady = true;
                this.updateBackendStatus(true);
                this.updateNPUStatus(response.data.npu_available);
                console.log('Backend is ready');
            } else {
                this.isBackendReady = false;
                this.updateBackendStatus(false);
                console.log('Backend not ready, status:', response.status);
            }
            
        } catch (error) {
            this.isBackendReady = false;
            this.updateBackendStatus(false);
            console.error('Backend connection failed:', error.message);
            console.error('Full error:', error);
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
            const response = await this.makeRequest(`${this.backendUrl}/api/ocr`, {
                method: 'POST',
                data: {
                    image_b64: imageData
                }
            });
            
            console.log('OCR response:', response.data);
            
            if (response.data.success) {
                this.currentText = response.data.text;
                console.log('Extracted text:', this.currentText);
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

    async handleFileUpload(event) {
        const file = event.target.files[0];
        if (!file) return;

        if (!this.isBackendReady) {
            this.showToast('Backend not ready. Please wait...', 'error');
            return;
        }

        try {
            this.showLoading('Processing uploaded image...');
            
            // Convert file to base64
            const reader = new FileReader();
            reader.onload = async (e) => {
                const imageData = e.target.result;
                await this.processImage(imageData);
            };
            reader.readAsDataURL(file);
            
        } catch (error) {
            console.error('File upload error:', error);
            this.showToast('File upload failed: ' + error.message, 'error');
        } finally {
            this.hideLoading();
        }
    }

    async processImage(imageData) {
        try {
            // Send to backend
            const response = await this.makeRequest(`${this.backendUrl}/api/ocr`, {
                method: 'POST',
                data: {
                    image_b64: imageData
                }
            });
            
            console.log('OCR response (file upload):', response.data);
            
            if (response.data.success) {
                this.currentText = response.data.text;
                console.log('Extracted text (file upload):', this.currentText);
                this.displayText(response.data);
                this.speakText(response.data.text);
                this.showToast('Text extracted successfully!', 'success');
            } else {
                throw new Error(response.data.error || 'OCR failed');
            }
            
        } catch (error) {
            console.error('OCR processing error:', error);
            this.showToast(`OCR failed: ${error.message}`, 'error');
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
                    const response = await this.makeRequest(`${this.backendUrl}/api/stt`, {
                        method: 'POST',
                        data: {
                            audio: audioData
                        }
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
            
            const response = await this.makeRequest(`${this.backendUrl}/api/command`, {
                method: 'POST',
                data: {
                    text: command
                }
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
            await this.makeRequest(`${this.backendUrl}/api/command`, {
                method: 'POST',
                data: {
                    text: text
                }
            });
        } catch (error) {
            console.error('TTS error:', error);
            this.showToast('Text-to-speech failed', 'error');
        }
    }

    displayText(ocrResult) {
        const textDisplay = document.getElementById('text-display');
        const metadata = document.getElementById('text-metadata');
        
        console.log('Displaying OCR result:', ocrResult);
        console.log('Text display element:', textDisplay);
        console.log('Metadata element:', metadata);
        
        if (!textDisplay) {
            console.error('Text display element not found!');
            return;
        }
        
        if (!metadata) {
            console.error('Metadata element not found!');
            return;
        }
        
        // Display extracted text with proper formatting
        if (ocrResult.text) {
            // Replace newlines with <br> tags and preserve formatting
            const formattedText = ocrResult.text.replace(/\n/g, '<br>');
            console.log('Formatted text:', formattedText);
            textDisplay.innerHTML = `<p>${formattedText}</p>`;
            textDisplay.classList.remove('placeholder');
            console.log('Text display innerHTML set to:', textDisplay.innerHTML);
        } else {
            textDisplay.innerHTML = '<p class="placeholder">No text detected</p>';
            console.log('No text detected, showing placeholder');
        }
        
        // Display metadata
        const inferenceTime = ocrResult.inference_time ? `${ocrResult.inference_time.toFixed(1)}ms` : 'N/A';
        const provider = ocrResult.npu_used ? 'NPU' : 'CPU';
        const confidence = ocrResult.confidence ? `${(ocrResult.confidence * 100).toFixed(1)}%` : 'N/A';
        
        metadata.innerHTML = `
            <div>Provider: ${provider} | Time: ${inferenceTime} | Confidence: ${confidence}</div>
        `;
        
        console.log('Text display completed');
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
