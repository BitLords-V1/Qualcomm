"""
Development runner for iLumina
Starts both backend and frontend in development mode
"""

import os
import sys
import subprocess
import time
import signal
import threading
from pathlib import Path

class ILuminaDevRunner:
    def __init__(self):
        self.backend_process = None
        self.frontend_process = None
        self.running = False
    
    def start_backend(self):
        """Start the Flask backend"""
        print("Starting backend...")
        
        backend_dir = Path("backend")
        if not backend_dir.exists():
            print("❌ Backend directory not found")
            return False
        
        try:
            self.backend_process = subprocess.Popen(
                [sys.executable, "app.py"],
                cwd=backend_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Wait for backend to start
            time.sleep(3)
            
            if self.backend_process.poll() is None:
                print("✅ Backend started successfully")
                return True
            else:
                print("❌ Backend failed to start")
                if self.backend_process.stdout:
                    print("Backend stdout:", self.backend_process.stdout.read())
                if self.backend_process.stderr:
                    print("Backend stderr:", self.backend_process.stderr.read())
                return False
                
        except Exception as e:
            print(f"❌ Failed to start backend: {e}")
            return False
    
    def start_frontend(self):
        """Start the Electron frontend"""
        print("Starting frontend...")
        
        frontend_dir = Path("frontend")
        if not frontend_dir.exists():
            print("❌ Frontend directory not found")
            return False
        
        try:
            self.frontend_process = subprocess.Popen(
                ["cmd.exe", "/c", os.path.join("node_modules", ".bin", "electron.cmd"), "."],
                cwd=frontend_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            print("✅ Frontend started successfully")
            return True
            
        except Exception as e:
            print(f"❌ Failed to start frontend: {e}")
            return False
    
    def monitor_processes(self):
        """Monitor backend and frontend processes"""
        while self.running:
            # Check backend
            if self.backend_process and self.backend_process.poll() is not None:
                print("❌ Backend process stopped unexpectedly")
                self.running = False
                break
            
            # Check frontend
            if self.frontend_process and self.frontend_process.poll() is not None:
                print("❌ Frontend process stopped unexpectedly")
                self.running = False
                break
            
            time.sleep(1)
    
    def cleanup(self):
        """Clean up processes"""
        print("\nShutting down...")
        self.running = False
        
        if self.backend_process:
            self.backend_process.terminate()
            try:
                self.backend_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.backend_process.kill()
        
        if self.frontend_process:
            self.frontend_process.terminate()
            try:
                self.frontend_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.frontend_process.kill()
        
        print("✅ Cleanup completed")
    
    def run(self):
        """Run the development environment"""
        print("iLumina Development Runner")
        print("=" * 30)
        
        # Start backend
        if not self.start_backend():
            sys.exit(1)
        
        # Start frontend
        if not self.start_frontend():
            self.cleanup()
            sys.exit(1)
        
        # Set up signal handlers
        def signal_handler(signum, frame):
            self.cleanup()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Start monitoring
        self.running = True
        monitor_thread = threading.Thread(target=self.monitor_processes)
        monitor_thread.daemon = True
        monitor_thread.start()
        
        print("\n🎉 iLumina development environment is running!")
        print("Press Ctrl+C to stop")
        
        try:
            # Keep main thread alive
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        finally:
            self.cleanup()

def main():
    """Main entry point"""
    runner = ILuminaDevRunner()
    runner.run()

if __name__ == "__main__":
    main()
