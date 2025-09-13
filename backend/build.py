"""
Build script for iLumina Backend
Creates standalone executable using PyInstaller
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def build_backend():
    """Build backend executable using PyInstaller"""
    print("Building iLumina Backend...")
    
    # Get current directory
    backend_dir = Path(__file__).parent
    dist_dir = backend_dir / "dist"
    
    # Clean previous builds
    if dist_dir.exists():
        shutil.rmtree(dist_dir)
    
    # PyInstaller command
    cmd = [
        "pyinstaller",
        "--onefile",
        "--name", "backend",
        "--distpath", str(dist_dir),
        "--workpath", str(backend_dir / "build"),
        "--specpath", str(backend_dir),
        "--add-data", "models;models",  # Include models directory
        "--hidden-import", "onnxruntime",
        "--hidden-import", "cv2",
        "--hidden-import", "PIL",
        "--hidden-import", "pyttsx3",
        "--hidden-import", "pyaudio",
        "--hidden-import", "wave",
        "app.py"
    ]
    
    try:
        # Run PyInstaller
        result = subprocess.run(cmd, cwd=backend_dir, check=True, capture_output=True, text=True)
        print("Backend build successful!")
        print(f"Executable created: {dist_dir / 'backend.exe'}")
        
        # Copy models to dist directory
        models_src = backend_dir / "models"
        models_dst = dist_dir / "models"
        
        if models_src.exists():
            shutil.copytree(models_src, models_dst)
            print("Models directory copied to dist")
        else:
            print("Warning: Models directory not found. Please add ONNX models to backend/models/")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"Build failed: {e}")
        print(f"Error output: {e.stderr}")
        return False
    except Exception as e:
        print(f"Unexpected error: {e}")
        return False

def install_dependencies():
    """Install required dependencies"""
    print("Installing dependencies...")
    
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
        print("Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Failed to install dependencies: {e}")
        return False

if __name__ == "__main__":
    print("iLumina Backend Build Script")
    print("=" * 40)
    
    # Install dependencies first
    if not install_dependencies():
        sys.exit(1)
    
    # Build backend
    if build_backend():
        print("\n✅ Backend build completed successfully!")
        print("\nNext steps:")
        print("1. Add ONNX models to backend/models/ directory")
        print("2. Test the backend executable")
        print("3. Build the frontend with 'npm run build'")
    else:
        print("\n❌ Backend build failed!")
        sys.exit(1)
