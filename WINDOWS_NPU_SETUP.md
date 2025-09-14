# 🚀 AI Exam Helper - Windows NPU Setup Guide

## 🎯 **WINNING HACKATHON IMPLEMENTATION**

This guide ensures your AI Exam Helper works perfectly on **Windows with Snapdragon X Elite NPU**.

---

## 📋 **Prerequisites**

### 1. **Hardware Requirements**
- ✅ Windows 11 PC with Snapdragon X Elite processor
- ✅ NPU (Neural Processing Unit) enabled
- ✅ At least 8GB RAM
- ✅ Internet connection for initial setup

### 2. **Software Requirements**
- ✅ Python 3.8+ installed
- ✅ Git installed
- ✅ AnythingLLM NPU version

---

## 🔧 **Step 1: Install AnythingLLM NPU**

### Download AnythingLLM NPU Version
1. Go to: https://useanything.com/
2. **IMPORTANT**: Download the **"AnythingLLM NPU"** version (not regular AnythingLLM)
3. Install and run AnythingLLM
4. Verify NPU is detected in the interface

### Configure LLM Provider
1. Open AnythingLLM
2. Go to **Settings** → **LLM Provider**
3. Select **"AnythingLLM NPU"** (not regular LLM providers)
4. Choose **"Llama 3.1 8B Chat 8K"** model
5. **Enable NPU acceleration** ✅
6. Set precision to **"int8"** for optimal NPU performance
7. Save configuration

---

## 🏗️ **Step 2: Create Workspace**

### Create Exam Helper Workspace
1. In AnythingLLM, click **"New Workspace"**
2. Name: **"exam-helper"**
3. Slug: **"exam-helper"** (must match exactly)
4. **Enable vision capabilities** ✅ (for PDF processing)
5. **Enable chat mode** ✅
6. Save workspace

### Generate API Key
1. Go to **Settings** → **Tools** → **Developer API**
2. Click **"Generate New API Key"**
3. Copy the API key (starts with `sk-` or similar)
4. **Keep this key secure!**

---

## ⚙️ **Step 3: Configure AI Exam Helper**

### Update Configuration File
1. Open `backend/config.yaml`
2. Replace `"your-api-key-here"` with your actual API key
3. Verify these settings:
   ```yaml
   api_key: "sk-your-actual-api-key-here"
   model_server_base_url: "http://localhost:3001/api/v1"
   workspace_slug: "exam-helper"
   stream: false
   npu_enabled: true
   ```

### Install Dependencies
```bash
cd backend
pip install -r requirements.txt
```

---

## 🚀 **Step 4: Run the Application**

### Start Backend
```bash
cd backend
python app.py
```

**Expected Output:**
```
INFO:__main__:✅ Configuration loaded
INFO:__main__:✅ AnythingLLM NPU connected!
INFO:__main__:🚀 Starting AI Exam Helper...
INFO:__main__:AnythingLLM Status: ✅ Connected
 * Running on http://127.0.0.1:5000
```

### Start Frontend
```bash
cd frontend
npm start
```

---

## 🧪 **Step 5: Test NPU Functionality**

### Test 1: Health Check
- Open browser to `http://localhost:5000/health`
- Should show: `"anythingllm_connected": true`
- Should show: `"npu_available": true`

### Test 2: PDF Processing
1. Upload a real exam PDF
2. Check logs for: `"🤖 Using AnythingLLM NPU for exam analysis..."`
3. Verify questions are extracted from actual PDF content
4. **NO hardcoded data should appear**

### Test 3: Student Interaction
1. Say "ready" → Should get AI response
2. Say "repeat" → Should get AI response
3. Say "slow" → Should get AI response
4. Answer a question → Should get AI feedback

---

## 🔍 **Troubleshooting**

### Issue: "AnythingLLM not available"
**Solution:**
1. Ensure AnythingLLM NPU is running
2. Check API key is correct
3. Verify workspace slug is "exam-helper"
4. Check port 3001 is not blocked

### Issue: "NPU not detected"
**Solution:**
1. Update Windows drivers
2. Enable NPU in BIOS
3. Check Windows Device Manager for NPU
4. Restart AnythingLLM

### Issue: "Questions not extracted"
**Solution:**
1. Ensure PDF has readable text (not scanned images)
2. Check AnythingLLM vision capabilities are enabled
3. Verify workspace has correct permissions

---

## 🎯 **Expected Behavior on Windows NPU**

### ✅ **What Should Work:**
- **Real PDF text extraction** using PyMuPDF
- **AI-powered question analysis** using AnythingLLM NPU
- **Intelligent student responses** using NPU-accelerated LLM
- **Dynamic question navigation** with AI assistance
- **Encouraging feedback** tailored for dyslexia support

### ❌ **What Should NOT Happen:**
- Hardcoded sample questions
- "Demo mode" messages
- Fallback responses
- Generic AI responses

---

## 🏆 **Hackathon Success Checklist**

- [ ] AnythingLLM NPU version installed and running
- [ ] NPU acceleration enabled and detected
- [ ] API key configured correctly
- [ ] Workspace "exam-helper" created
- [ ] Backend shows "✅ AnythingLLM NPU connected!"
- [ ] Frontend shows "AI Assistant: ✅ Connected"
- [ ] PDF upload extracts real questions
- [ ] Student interactions get AI responses
- [ ] No hardcoded data appears
- [ ] All functionality works with real AI

---

## 🚨 **Critical Success Factors**

1. **Use AnythingLLM NPU version** (not regular AnythingLLM)
2. **Enable NPU acceleration** in settings
3. **Correct API key** in config.yaml
4. **Exact workspace slug** "exam-helper"
5. **Real PDF processing** (no hardcoded data)

---

## 📞 **Support**

If you encounter issues:
1. Check AnythingLLM logs for NPU status
2. Verify API key permissions
3. Test with simple PDF first
4. Ensure all dependencies are installed

**Your AI Exam Helper is now ready to win the hackathon! 🏆**
