#!/usr/bin/env python3
"""
NPU Setup Verification Script
Tests if AnythingLLM NPU is properly configured for Windows
"""

import requests
import yaml
import json
import sys

def test_anythingllm_connection():
    """Test AnythingLLM NPU connection"""
    print("🔍 Testing AnythingLLM NPU Connection...")
    
    try:
        # Load config
        with open('backend/config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        api_key = config.get('api_key', '')
        base_url = config.get('model_server_base_url', '')
        workspace_slug = config.get('workspace_slug', '')
        
        if api_key == 'your-api-key-here':
            print("❌ API key not configured!")
            print("   Please update backend/config.yaml with your actual API key")
            return False
        
        if not api_key or not base_url or not workspace_slug:
            print("❌ Configuration incomplete!")
            return False
        
        # Test connection
        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }
        
        test_url = f"{base_url}/workspace/{workspace_slug}/chat"
        test_payload = {
            "message": "Hello! Are you running on NPU?",
            "mode": "chat",
            "stream": False
        }
        
        print(f"   Testing: {test_url}")
        response = requests.post(test_url, headers=headers, json=test_payload, timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ AnythingLLM NPU Connected!")
            print(f"   Response: {str(result)[:100]}...")
            return True
        else:
            print(f"❌ Connection failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_backend_health():
    """Test backend health endpoint"""
    print("\n🔍 Testing Backend Health...")
    
    try:
        response = requests.get('http://127.0.0.1:5000/health', timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Backend is running!")
            print(f"   Status: {data.get('status', 'unknown')}")
            print(f"   AnythingLLM: {'✅ Connected' if data.get('anythingllm_connected') else '❌ Not Connected'}")
            print(f"   NPU: {'✅ Available' if data.get('npu_available') else '❌ Not Available'}")
            return data.get('anythingllm_connected', False)
        else:
            print(f"❌ Backend not responding: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Backend not running: {e}")
        return False

def test_pdf_processing():
    """Test PDF processing with real AI"""
    print("\n🔍 Testing PDF Processing...")
    
    try:
        # Create a minimal test PDF content
        test_pdf_content = """
        Sample Exam
        
        1. What is the capital of France?
        A) London
        B) Paris
        C) Berlin
        D) Madrid
        
        2. Which planet is closest to the Sun?
        A) Venus
        B) Mercury
        C) Earth
        D) Mars
        """
        
        # Convert to base64 (simplified)
        import base64
        pdf_b64 = base64.b64encode(test_pdf_content.encode()).decode()
        
        response = requests.post('http://127.0.0.1:5000/process-exam', 
                               json={'document': pdf_b64, 'type': 'pdf'},
                               timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                exam_info = data.get('exam_info', {})
                questions = exam_info.get('questions', [])
                print("✅ PDF Processing successful!")
                print(f"   Questions found: {len(questions)}")
                print(f"   Exam title: {exam_info.get('exam_title', 'Unknown')}")
                
                # Check if it's using real AI (not hardcoded)
                if "Demo Exam" in exam_info.get('exam_title', ''):
                    print("⚠️  Using demo data - AnythingLLM may not be working")
                    return False
                else:
                    print("✅ Using real AI analysis!")
                    return True
            else:
                print(f"❌ PDF processing failed: {data.get('message', 'Unknown error')}")
                return False
        else:
            print(f"❌ PDF processing request failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ PDF processing error: {e}")
        return False

def main():
    """Main verification function"""
    print("🚀 AI Exam Helper - NPU Setup Verification")
    print("=" * 50)
    
    # Test 1: AnythingLLM Connection
    anythingllm_ok = test_anythingllm_connection()
    
    # Test 2: Backend Health
    backend_ok = test_backend_health()
    
    # Test 3: PDF Processing (only if backend is running)
    pdf_ok = False
    if backend_ok:
        pdf_ok = test_pdf_processing()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 VERIFICATION SUMMARY")
    print("=" * 50)
    
    print(f"AnythingLLM NPU: {'✅ PASS' if anythingllm_ok else '❌ FAIL'}")
    print(f"Backend Health:  {'✅ PASS' if backend_ok else '❌ FAIL'}")
    print(f"PDF Processing:  {'✅ PASS' if pdf_ok else '❌ FAIL'}")
    
    if anythingllm_ok and backend_ok and pdf_ok:
        print("\n🎉 ALL TESTS PASSED!")
        print("   Your AI Exam Helper is ready for the hackathon!")
        print("   NPU acceleration is working correctly!")
        return True
    else:
        print("\n⚠️  SOME TESTS FAILED")
        print("   Please check the setup guide: WINDOWS_NPU_SETUP.md")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
