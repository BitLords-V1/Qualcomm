"""
Setup script for iLumina
Installs dependencies and sets up the development environment
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def run_command(command, cwd=None):
    """Run a command and return success status"""
    try:
        result = subprocess.run(
            command, 
            cwd=cwd, 
            shell=True, 
            check=True, 
            capture_output=True, 
            text=True
        )
        print(f"✅ {command}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {command}")
        print(f"Error: {e.stderr}")
        return False

def check_system():
    """Check system requirements"""
    print("Checking system requirements...")
    
    # Check OS
    if platform.system() != "Windows":
        print("❌ iLumina requires Windows-on-Snapdragon")
        return False
    
    # Check architecture
    if platform.machine() != "ARM64":
        print("⚠️  Warning: Not running on ARM64. iLumina is optimized for Windows-on-Snapdragon")
    
    print(f"✅ System: {platform.system()} {platform.machine()}")
    return True

def install_python_dependencies():
    """Install Python dependencies"""
    print("\nInstalling Python dependencies...")
    
    backend_dir = Path("backend")
    if not backend_dir.exists():
        print("❌ Backend directory not found")
        return False
    
    return run_command(f"{sys.executable} -m pip install -r requirements.txt", cwd=backend_dir)

def install_node_dependencies():
    """Install Node.js dependencies"""
    print("\nInstalling Node.js dependencies...")
    
    frontend_dir = Path("frontend")
    if not frontend_dir.exists():
        print("❌ Frontend directory not found")
        return False
    
    return run_command("npm install", cwd=frontend_dir)

def create_directories():
    """Create necessary directories"""
    print("\nCreating directories...")
    
    directories = [
        "backend/models",
        "backend/dist",
        "backend/build",
        "frontend/dist",
        "frontend/build",
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created: {directory}")
    
    return True

def check_models():
    """Check if models are present"""
    print("\nChecking models...")
    
    models_dir = Path("backend/models")
    required_models = [
        "easyocr_qnn.onnx",
        "whisper_tiny_en_qnn.onnx"
    ]
    
    missing_models = []
    for model in required_models:
        model_path = models_dir / model
        if not model_path.exists():
            missing_models.append(model)
        else:
            print(f"✅ Found: {model}")
    
    if missing_models:
        print(f"⚠️  Missing models: {', '.join(missing_models)}")
        print("Please download models from Qualcomm AI Hub and place them in backend/models/")
        return False
    
    return True

def main():
    """Main setup process"""
    print("iLumina Setup Script")
    print("=" * 30)
    
    # Check system
    if not check_system():
        sys.exit(1)
    
    # Create directories
    if not create_directories():
        sys.exit(1)
    
    # Install Python dependencies
    if not install_python_dependencies():
        print("❌ Failed to install Python dependencies")
        sys.exit(1)
    
    # Install Node.js dependencies
    if not install_node_dependencies():
        print("❌ Failed to install Node.js dependencies")
        sys.exit(1)
    
    # Check models
    check_models()
    
    print("\n🎉 iLumina setup completed!")
    print("\nNext steps:")
    print("1. Add ONNX models to backend/models/ directory")
    print("2. Run 'python build.py' to build the application")
    print("3. Test with 'npm start' in frontend directory")

if __name__ == "__main__":
    main()
