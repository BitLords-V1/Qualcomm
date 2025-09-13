# iLumina Installation Guide

Complete installation guide for iLumina on Windows-on-Snapdragon devices.

## Prerequisites

### System Requirements
- **OS**: Windows 11 on Snapdragon (ARM64)
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 2GB free space
- **NPU**: Qualcomm Hexagon NPU (for optimal performance)

### Required Software
- **Python 3.9+** (ARM64 version)
- **Node.js 18+** (ARM64 version)
- **Git** (for cloning the repository)

## Installation Methods

### Method 1: Quick Setup (Recommended)

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/ilumina.git
   cd ilumina
   ```

2. **Run the setup script**:
   ```bash
   python setup.py
   ```

3. **Add ONNX models**:
   - Download models from [Qualcomm AI Hub](https://aihub.qualcomm.com)
   - Place `easyocr_qnn.onnx` and `whisper_tiny_en_qnn.onnx` in `backend/models/`

4. **Build the application**:
   ```bash
   python build.py
   ```

5. **Install the application**:
   - Run `frontend/dist/iLumina-Setup.exe`
   - Follow the installation wizard

### Method 2: Manual Setup

#### Backend Setup

1. **Navigate to backend directory**:
   ```bash
   cd backend
   ```

2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Add ONNX models**:
   - Download models from Qualcomm AI Hub
   - Place models in `models/` directory

4. **Build backend executable**:
   ```bash
   python build.py
   ```

#### Frontend Setup

1. **Navigate to frontend directory**:
   ```bash
   cd frontend
   ```

2. **Install Node.js dependencies**:
   ```bash
   npm install
   ```

3. **Build frontend**:
   ```bash
   npm run build
   ```

4. **Create installer**:
   ```bash
   npm run dist
   ```

## Model Setup

### Downloading Models

1. **Visit Qualcomm AI Hub**:
   - Go to [https://aihub.qualcomm.com](https://aihub.qualcomm.com)
   - Create an account if needed

2. **Download EasyOCR Model**:
   - Search for "EasyOCR"
   - Download the QNN-compiled version
   - Rename to `easyocr_qnn.onnx`

3. **Download Whisper Model**:
   - Search for "Whisper Tiny EN"
   - Download the QNN-compiled version
   - Rename to `whisper_tiny_en_qnn.onnx`

4. **Place Models**:
   ```
   backend/
   └── models/
       ├── easyocr_qnn.onnx
       └── whisper_tiny_en_qnn.onnx
   ```

### Model Verification

Run the test script to verify models are working:
```bash
python test_backend.py
```

## Development Setup

### Running in Development Mode

1. **Start backend**:
   ```bash
   cd backend
   python app.py
   ```

2. **Start frontend** (in new terminal):
   ```bash
   cd frontend
   npm start
   ```

3. **Or use the development runner**:
   ```bash
   python run_dev.py
   ```

### Testing

1. **Test backend only**:
   ```bash
   python test_backend.py
   ```

2. **Test full application**:
   - Start the application
   - Test camera capture
   - Test voice commands
   - Verify NPU usage

## Troubleshooting

### Common Issues

#### 1. NPU Not Detected
**Symptoms**: Status shows "CPU Fallback" instead of "NPU Active"

**Solutions**:
- Install Qualcomm QNN drivers
- Verify Windows-on-Snapdragon device
- Check ONNX Runtime QNN provider installation

#### 2. Models Not Found
**Symptoms**: Error messages about missing model files

**Solutions**:
- Verify models are in `backend/models/` directory
- Check file names match exactly (case-sensitive)
- Ensure models are QNN-compiled, not CPU-only

#### 3. Camera Access Denied
**Symptoms**: Camera preview not working

**Solutions**:
- Grant camera permissions in Windows Settings
- Check camera is not being used by another application
- Verify camera drivers are installed

#### 4. Microphone Access Denied
**Symptoms**: Voice recording not working

**Solutions**:
- Grant microphone permissions in Windows Settings
- Check microphone is not being used by another application
- Verify audio drivers are installed

#### 5. Backend Connection Failed
**Symptoms**: Frontend shows "Backend Offline"

**Solutions**:
- Check if backend is running on port 5000
- Verify firewall settings
- Check for port conflicts

### Performance Issues

#### Slow Inference
- Ensure NPU is being used (check status indicator)
- Verify models are QNN-compiled
- Check system resources (CPU, RAM)

#### High Memory Usage
- Close other applications
- Restart the application
- Check for memory leaks

### Getting Help

1. **Check logs**:
   - Backend logs in console output
   - Frontend logs in Developer Tools (F12)

2. **Verify installation**:
   - Run `python setup.py` to check dependencies
   - Run `python test_backend.py` to test backend

3. **Contact support**:
   - Check GitHub Issues
   - Contact Qualcomm support for NPU issues

## Uninstallation

### Remove Application
1. Use Windows "Add or Remove Programs"
2. Search for "iLumina"
3. Click "Uninstall"

### Clean Up Files
1. Delete installation directory
2. Remove any remaining files in:
   - `%APPDATA%/iLumina`
   - `%LOCALAPPDATA%/iLumina`

## Advanced Configuration

### Environment Variables
- `ILUMINA_DEBUG`: Enable debug mode
- `ILUMINA_LOG_LEVEL`: Set log level (DEBUG, INFO, WARNING, ERROR)
- `ILUMINA_BACKEND_PORT`: Change backend port (default: 5000)

### Custom Models
- Place custom ONNX models in `backend/models/`
- Update model paths in configuration files
- Ensure models are QNN-compiled

### Performance Tuning
- Adjust NPU memory allocation
- Configure ONNX Runtime providers
- Optimize model parameters

## Support

For additional help:
- **Documentation**: Check README.md and INSTALL.md
- **Issues**: Report bugs on GitHub Issues
- **Qualcomm Support**: Contact for NPU-related issues
- **Community**: Join Qualcomm Developer Forums
