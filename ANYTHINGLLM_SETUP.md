# AI Exam Helper - AnythingLLM Setup Guide
# WINNING HACKATHON IMPLEMENTATION

## 🏆 **COMPLETE SETUP GUIDE FOR WINNING THE HACKATHON**

This guide will help you set up AnythingLLM with NPU acceleration for your AI Exam Helper application.

## 📋 **PREREQUISITES**

- Windows 11 on Snapdragon X Elite (Copilot+ PC)
- Admin privileges for installation
- Internet connection for initial setup

## 🚀 **STEP 1: Install AnythingLLM**

### Option A: Direct Download (Recommended)
1. Go to [https://useanything.com/](https://useanything.com/)
2. Download **"AnythingLLM NPU"** version for Windows ARM64
3. Run the installer as Administrator
4. Follow the installation wizard

### Option B: Alternative Download
1. Visit [https://github.com/whatif-dev/chat-anything-llm/releases](https://github.com/whatif-dev/chat-anything-llm/releases)
2. Download the latest Windows ARM64 release
3. Extract and run `AnythingLLM.exe`

## ⚙️ **STEP 2: Configure AnythingLLM**

### 2.1 Initial Setup
1. Launch AnythingLLM
2. Create your first workspace:
   - Name: `exam-helper`
   - Slug: `exam-helper`
   - Description: `AI Exam Helper for Accessibility`

### 2.2 Configure LLM Provider
1. Go to **Settings** → **LLM Provider**
2. Select **"AnythingLLM NPU"**
3. Choose **"Llama 3.1 8B Chat 8K"** model
4. Enable **NPU acceleration**
5. Set precision to **INT8** (optimized for NPU)
6. Save configuration

### 2.3 Enable Vision Capabilities
1. Go to **Settings** → **Workspace Settings**
2. Enable **"Vision Processing"**
3. Enable **"Document Processing"**
4. Set **"Max Document Size"** to 50MB

## 🔑 **STEP 3: Generate API Key**

1. Go to **Settings** → **Tools** → **Developer API**
2. Click **"Generate New API Key"**
3. Copy the generated key
4. Update `backend/config.yaml` with your key:

```yaml
api_key: "your-actual-api-key-here"
```

## 🧪 **STEP 4: Test Installation**

### 4.1 Test AnythingLLM
1. Open AnythingLLM web interface
2. Go to your `exam-helper` workspace
3. Try sending a test message: "Hello, are you working?"
4. Verify you get a response

### 4.2 Test NPU Acceleration
1. Check **Settings** → **System** → **Hardware**
2. Verify **"NPU Acceleration"** shows as **"Active"**
3. Look for **"Snapdragon NPU"** in the hardware list

### 4.3 Test API Connection
1. Run your AI Exam Helper backend:
   ```bash
   cd backend
   python app.py
   ```
2. Check the console for: `✅ AnythingLLM NPU connection established!`
3. Visit `http://localhost:5000/health` to verify status

## 📁 **STEP 5: Configure Your Application**

### 5.1 Update Configuration
Edit `backend/config.yaml`:

```yaml
# Replace these values with your actual setup
api_key: "your-actual-api-key-here"
model_server_base_url: "http://localhost:3001/api/v1"
workspace_slug: "exam-helper"
stream: true
stream_timeout: 60

# NPU Optimization Settings
npu_enabled: true
model_precision: "int8"
batch_size: 1
max_tokens: 2048

# Exam Helper Settings
reading_speed: "slow"
voice_enabled: true
accessibility_mode: true
dyslexia_support: true
```

### 5.2 Test Exam Processing
1. Start your application
2. Upload a sample PDF exam
3. Verify the AI processes and extracts questions
4. Test the interactive exam session

## 🎯 **STEP 6: Hackathon Optimization**

### 6.1 Performance Tuning
1. **Close unnecessary applications** to free up NPU resources
2. **Set AnythingLLM priority** to High in Task Manager
3. **Disable Windows Defender real-time scanning** temporarily
4. **Use wired internet** for stable connection

### 6.2 Demo Preparation
1. **Prepare sample exam PDFs** with clear questions
2. **Test voice commands** (repeat, slower, etc.)
3. **Practice the demo flow**:
   - Upload exam → Process → Start session → Answer questions → Summary
4. **Prepare backup plans** if AnythingLLM has issues

## 🔧 **TROUBLESHOOTING**

### Common Issues:

#### "AnythingLLM not connected"
- **Solution**: Check if AnythingLLM is running on port 3001
- **Check**: `http://localhost:3001` in browser

#### "API Key invalid"
- **Solution**: Regenerate API key in AnythingLLM settings
- **Check**: Copy-paste the key correctly in config.yaml

#### "NPU not detected"
- **Solution**: Restart AnythingLLM and your application
- **Check**: Windows Device Manager for Snapdragon NPU

#### "Model loading failed"
- **Solution**: Re-download the Llama 3.1 8B model
- **Check**: Available disk space (need ~8GB)

### Performance Issues:

#### Slow responses
- **Solution**: Reduce `max_tokens` in config.yaml
- **Check**: Close other NPU-intensive applications

#### Memory issues
- **Solution**: Reduce `batch_size` to 1
- **Check**: Available RAM (need at least 8GB free)

## 📊 **VERIFICATION CHECKLIST**

Before the hackathon, verify:

- [ ] AnythingLLM launches successfully
- [ ] NPU acceleration is active
- [ ] API key works (test with curl or Postman)
- [ ] Your app connects to AnythingLLM
- [ ] Exam processing works with sample PDF
- [ ] Voice commands work (repeat, slower)
- [ ] Exam session flows correctly
- [ ] Summary generation works
- [ ] Performance is acceptable (< 5 seconds per question)

## 🏆 **WINNING TIPS**

1. **Demo Flow**: Practice the complete user journey
2. **Backup Plan**: Have a working version without AnythingLLM
3. **Performance**: Show NPU utilization in Task Manager
4. **Accessibility**: Emphasize the dyslexia support features
5. **Innovation**: Highlight the on-device AI processing

## 📞 **SUPPORT**

If you encounter issues:
1. Check AnythingLLM logs in `%APPDATA%/AnythingLLM/logs/`
2. Check your app logs in `backend/logs/`
3. Restart both AnythingLLM and your application
4. Verify all configuration values

## 🎉 **YOU'RE READY TO WIN!**

With this setup, your AI Exam Helper will:
- ✅ Use NPU acceleration for fast AI processing
- ✅ Process exam documents with vision AI
- ✅ Provide accessible reading assistance
- ✅ Track student progress and answers
- ✅ Generate intelligent summaries
- ✅ Work completely on-device for privacy

**Good luck at the hackathon! 🚀**
