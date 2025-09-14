"""
Master build script for iLumina
Builds both backend and frontend, creates complete installer
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def run_command(command, cwd=None, shell=True):
    """Run a command and return success status"""
    try:
        result = subprocess.run(
            command, 
            cwd=cwd, 
            shell=shell, 
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

def check_prerequisites():
    """Check if all required tools are installed"""
    print("Checking prerequisites...")
    
    # Check Python
    if not run_command("python --version"):
        print("Python is required but not found")
        return False
    
    # Check Node.js
    if not run_command("node --version"):
        print("Node.js is required but not found")
        return False
    
    # Check npm
    if not run_command("npm --version"):
        print("npm is required but not found")
        return False
    
    print("✅ All prerequisites found")
    return True

def build_backend():
    """Build the backend executable"""
    print("\nBuilding backend...")
    
    backend_dir = Path("backend")
    if not backend_dir.exists():
        print("❌ Backend directory not found")
        return False
    
    # Install Python dependencies
    if not run_command("pip install -r requirements.txt", cwd=backend_dir):
        print("❌ Failed to install Python dependencies")
        return False
    
    # Build backend
    if not run_command("python build.py", cwd=backend_dir):
        print("❌ Failed to build backend")
        return False
    
    print("✅ Backend built successfully")
    return True

def build_frontend():
    """Build the frontend Electron app"""
    print("\nBuilding frontend...")
    
    frontend_dir = Path("frontend")
    if not frontend_dir.exists():
        print("❌ Frontend directory not found")
        return False
    
    # Install Node.js dependencies
    if not run_command("npm install", cwd=frontend_dir):
        print("❌ Failed to install Node.js dependencies")
        return False
    
    # Build frontend
    if not run_command("npm run build", cwd=frontend_dir):
        print("❌ Failed to build frontend")
        return False
    
    print("✅ Frontend built successfully")
    return True

def create_installer():
    """Create the final installer"""
    print("\nCreating installer...")
    
    frontend_dir = Path("frontend")
    
    # Create installer
    if not run_command("npm run dist", cwd=frontend_dir):
        print("❌ Failed to create installer")
        return False
    
    print("✅ Installer created successfully")
    return True

def verify_build():
    """Verify the build output"""
    print("\nVerifying build...")
    
    # Check backend executable
    backend_exe = Path("backend/dist/backend.exe")
    if not backend_exe.exists():
        print("❌ Backend executable not found")
        return False
    
    # Check frontend installer
    installer_dir = Path("frontend/dist")
    if not installer_dir.exists():
        print("❌ Frontend dist directory not found")
        return False
    
    # Look for installer file
    installer_files = list(installer_dir.glob("*.exe"))
    if not installer_files:
        print("❌ Installer file not found")
        return False
    
    print(f"✅ Installer found: {installer_files[0].name}")
    return True

def main():
    """Main build process"""
    print("iLumina Master Build Script")
    print("=" * 50)
    
    # Check prerequisites
    if not check_prerequisites():
        print("\n❌ Prerequisites check failed")
        sys.exit(1)
    
    # Build backend
    if not build_backend():
        print("\n❌ Backend build failed")
        sys.exit(1)
    
    # Build frontend
    if not build_frontend():
        print("\n❌ Frontend build failed")
        sys.exit(1)
    
    # Create installer
    if not create_installer():
        print("\n❌ Installer creation failed")
        sys.exit(1)
    
    # Verify build
    if not verify_build():
        print("\n❌ Build verification failed")
        sys.exit(1)
    
    print("\n🎉 iLumina build completed successfully!")
    print("\nNext steps:")
    print("1. Add ONNX models to backend/models/ directory")
    print("2. Test the installer: frontend/dist/iLumina-Setup.exe")
    print("3. Distribute the installer to users")

if __name__ == "__main__":
    main()
