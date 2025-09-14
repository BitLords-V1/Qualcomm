/**
 * iLumina PDF Upload Frontend - MCQ Exam Helper
 * Handles PDF upload and MCQ question extraction
 */

const { ipcRenderer } = require('electron');
const axios = require('axios');

class ILuminaPDFRenderer {
    constructor() {
        this.backendUrl = 'http://127.0.0.1:5000';
        this.isBackendReady = false;
        this.currentQuestions = '';
        this.currentText = '';
        this.isSpeaking = false;
        
        // Initialize the application
        this.init();
    }

    async init() {
        console.log('Initializing iLumina PDF renderer...');
        
        // Setup event listeners
        this.setupEventListeners();
        
        // Check backend status
        await this.checkBackendStatus();
        
        // Update UI
        this.updateUI();
        
        console.log('iLumina PDF renderer initialized successfully');
    }

    setupEventListeners() {
        // PDF upload controls
        document.getElementById('file-input').addEventListener('change', (event) => {
            this.handleFileSelect(event);
        });

        document.getElementById('upload-btn').addEventListener('click', () => {
            this.uploadAndProcessFile();
        });

        document.getElementById('clear-btn').addEventListener('click', () => {
            this.clearResults();
        });

        // Copy to clipboard
        document.getElementById('copy-btn').addEventListener('click', () => {
            this.copyToClipboard();
        });

        // TTS controls
        document.getElementById('speak-btn').addEventListener('click', () => {
            this.speakCurrentText();
        });

        document.getElementById('repeat-btn').addEventListener('click', () => {
            this.repeatText();
        });

        document.getElementById('stop-speech-btn').addEventListener('click', () => {
            this.stopSpeech();
        });
    }

    async checkBackendStatus() {
        try {
            const response = await axios.get(`${this.backendUrl}/health`);
            this.isBackendReady = response.data.ocr_ready;
            
            console.log('Backend status:', response.data);
            
            // Update TTS status
            this.updateTTSStatus(response.data.tts_ready);
            
            if (this.isBackendReady) {
                this.updateStatus('Backend ready! Upload a PDF to extract MCQ questions.', 'success');
            } else {
                this.updateStatus('Backend not ready. Please check server status.', 'error');
            }
        } catch (error) {
            console.error('Backend check failed:', error);
            this.isBackendReady = false;
            this.updateStatus('Cannot connect to backend. Please start the server.', 'error');
        }
    }

    handleFileSelect(event) {
        const file = event.target.files[0];
        if (file) {
            const isPDF = file.type === 'application/pdf';
            const isImage = file.type.startsWith('image/');
            
            if (isPDF || isImage) {
                this.updateStatus(`Selected: ${file.name}`, 'info');
                document.getElementById('upload-btn').disabled = false;
            } else {
                this.updateStatus('Please select a PDF or image file.', 'error');
                document.getElementById('upload-btn').disabled = true;
            }
        }
    }

    async uploadAndProcessFile() {
        const fileInput = document.getElementById('file-input');
        const file = fileInput.files[0];
        
        if (!file) {
            this.updateStatus('Please select a file first.', 'error');
            return;
        }

        if (!this.isBackendReady) {
            this.updateStatus('Backend not ready. Please check server status.', 'error');
            return;
        }

        try {
            this.updateStatus('Processing file... Please wait.', 'info');
            document.getElementById('upload-btn').disabled = true;
            document.getElementById('clear-btn').disabled = true;

            // Create FormData for file upload
            const formData = new FormData();
            formData.append('file', file);

            // Determine endpoint based on file type
            const isPDF = file.type === 'application/pdf';
            const endpoint = isPDF ? '/upload-pdf' : '/upload-image';

            // Upload and process file
            const response = await axios.post(`${this.backendUrl}${endpoint}`, formData, {
                headers: {
                    'Content-Type': 'multipart/form-data'
                },
                timeout: 120000 // 2 minutes timeout for large files
            });

            if (response.data.success) {
                this.currentQuestions = response.data.questions;
                this.currentText = response.data.extracted_text || '';
                this.displayQuestions(response.data);
                this.updateStatus(`Successfully processed ${response.data.filename}!`, 'success');
                
                // Show TTS controls if text was extracted
                if (this.currentText.trim()) {
                    this.showTTSControls();
                    this.updateTTSStatus(true);
                }
            } else {
                this.updateStatus(`Error: ${response.data.error}`, 'error');
            }

        } catch (error) {
            console.error('File processing error:', error);
            
            if (error.response) {
                this.updateStatus(`Error: ${error.response.data.error || error.response.statusText}`, 'error');
            } else if (error.code === 'ECONNABORTED') {
                this.updateStatus('Processing timeout. File might be too large or complex.', 'error');
            } else {
                this.updateStatus('Network error. Please check your connection.', 'error');
            }
        } finally {
            document.getElementById('upload-btn').disabled = false;
            document.getElementById('clear-btn').disabled = false;
        }
    }

    displayQuestions(data) {
        const questionsDiv = document.getElementById('questions-output');
        
        // Create HTML for questions
        let html = `
            <div class="results-header">
                <h3>📄 MCQ Questions Extracted</h3>
                <div class="stats">
                    <span class="stat">📄 Pages: ${data.pages_processed}</span>
                    <span class="stat">🔍 Text Regions: ${data.text_regions_found}</span>
                    <span class="stat">📁 File: ${data.filename}</span>
                </div>
            </div>
            <div class="questions-content">
                <pre>${this.currentQuestions}</pre>
            </div>
        `;
        
        questionsDiv.innerHTML = html;
        questionsDiv.style.display = 'block';
        
        // Enable copy button
        document.getElementById('copy-btn').disabled = false;
    }

    clearResults() {
        document.getElementById('questions-output').innerHTML = '';
        document.getElementById('questions-output').style.display = 'none';
        document.getElementById('file-input').value = '';
        document.getElementById('upload-btn').disabled = true;
        document.getElementById('copy-btn').disabled = true;
        this.currentQuestions = '';
        this.updateStatus('Results cleared. Upload a new PDF to extract questions.', 'info');
    }

    async copyToClipboard() {
        if (this.currentQuestions) {
            try {
                await navigator.clipboard.writeText(this.currentQuestions);
                this.updateStatus('Questions copied to clipboard!', 'success');
            } catch (error) {
                console.error('Copy failed:', error);
                this.updateStatus('Failed to copy to clipboard.', 'error');
            }
        }
    }

    updateStatus(message, type) {
        const statusDiv = document.getElementById('status');
        statusDiv.textContent = message;
        statusDiv.className = `status ${type}`;
        
        // Auto-hide success messages after 3 seconds
        if (type === 'success') {
            setTimeout(() => {
                statusDiv.textContent = '';
                statusDiv.className = 'status';
            }, 3000);
        }
    }

    updateUI() {
        // Update UI based on backend status
        const uploadBtn = document.getElementById('upload-btn');
        uploadBtn.disabled = !this.isBackendReady;
        
        if (!this.isBackendReady) {
            this.updateStatus('Waiting for backend to start...', 'info');
        }
    }

    // TTS Methods
    showTTSControls() {
        const ttsControls = document.getElementById('tts-controls');
        ttsControls.style.display = 'block';
    }

    hideTTSControls() {
        const ttsControls = document.getElementById('tts-controls');
        ttsControls.style.display = 'none';
    }

    updateTTSStatus(isAvailable, message = null) {
        const indicator = document.getElementById('tts-indicator');
        const messageEl = document.getElementById('tts-message');
        
        if (isAvailable) {
            indicator.textContent = '🔊';
            messageEl.textContent = message || 'TTS Ready';
        } else {
            indicator.textContent = '🔇';
            messageEl.textContent = 'TTS Not Available';
        }
    }

    async speakCurrentText() {
        if (!this.currentText.trim()) {
            this.updateStatus('No text to speak. Please upload a file first.', 'error');
            return;
        }

        try {
            this.isSpeaking = true;
            this.updateTTSStatus(true, 'Speaking...');
            
            const response = await axios.post(`${this.backendUrl}/speak`, {
                text: this.currentText
            });

            if (response.data.success) {
                this.updateStatus('Speaking text...', 'info');
            } else {
                this.updateStatus('TTS error: ' + response.data.error, 'error');
            }
        } catch (error) {
            console.error('TTS error:', error);
            this.updateStatus('Failed to speak text', 'error');
        } finally {
            this.isSpeaking = false;
            this.updateTTSStatus(true);
        }
    }

    async repeatText() {
        try {
            this.isSpeaking = true;
            this.updateTTSStatus(true, 'Repeating...');
            
            const response = await axios.post(`${this.backendUrl}/repeat`);

            if (response.data.success) {
                this.updateStatus('Repeating last text...', 'info');
            } else {
                this.updateStatus('Repeat error: ' + response.data.error, 'error');
            }
        } catch (error) {
            console.error('Repeat error:', error);
            this.updateStatus('Failed to repeat text', 'error');
        } finally {
            this.isSpeaking = false;
            this.updateTTSStatus(true);
        }
    }

    stopSpeech() {
        // Note: Backend doesn't have stop endpoint yet, but we can update UI
        this.isSpeaking = false;
        this.updateTTSStatus(true);
        this.updateStatus('Speech stopped', 'info');
    }
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new ILuminaPDFRenderer();
});