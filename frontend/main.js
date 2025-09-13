/**
 * iLumina Main Process - Electron Application Entry Point
 * Manages backend process and main window
 */

const { app, BrowserWindow, ipcMain, dialog, Menu } = require('electron');
const path = require('path');
const { spawn } = require('child_process');
const axios = require('axios');

class ILuminaApp {
    constructor() {
        this.mainWindow = null;
        this.backendProcess = null;
        this.backendPort = 5000;
        this.backendUrl = `http://127.0.0.1:${this.backendPort}`;
        this.isBackendReady = false;
    }

    async createWindow() {
        // Create the browser window
        this.mainWindow = new BrowserWindow({
            width: 1920,
            height: 1080,
            fullscreen: true,
            webPreferences: {
                nodeIntegration: true,
                contextIsolation: false,
                enableRemoteModule: true
            },
            icon: path.join(__dirname, 'assets', 'icon.png'),
            show: false,
            titleBarStyle: 'hidden',
            frame: false
        });

        // Load the app
        await this.mainWindow.loadFile('index.html');

        // Show window when ready
        this.mainWindow.once('ready-to-show', () => {
            this.mainWindow.show();
            this.setupMenu();
        });

        // Handle window closed
        this.mainWindow.on('closed', () => {
            this.cleanup();
        });

        // Setup IPC handlers
        this.setupIPC();
    }

    setupMenu() {
        // Create application menu
        const template = [
            {
                label: 'iLumina',
                submenu: [
                    {
                        label: 'About iLumina',
                        click: () => {
                            dialog.showMessageBox(this.mainWindow, {
                                type: 'info',
                                title: 'About iLumina',
                                message: 'iLumina - Offline Exam Helper',
                                detail: 'AI-powered exam assistance for Windows-on-Snapdragon\nPowered by Qualcomm NPU acceleration'
                            });
                        }
                    },
                    { type: 'separator' },
                    {
                        label: 'Exit',
                        accelerator: 'CmdOrCtrl+Q',
                        click: () => {
                            app.quit();
                        }
                    }
                ]
            },
            {
                label: 'View',
                submenu: [
                    {
                        label: 'Toggle Fullscreen',
                        accelerator: 'F11',
                        click: () => {
                            this.mainWindow.setFullScreen(!this.mainWindow.isFullScreen());
                        }
                    },
                    {
                        label: 'Reload',
                        accelerator: 'CmdOrCtrl+R',
                        click: () => {
                            this.mainWindow.reload();
                        }
                    },
                    {
                        label: 'Toggle Developer Tools',
                        accelerator: 'F12',
                        click: () => {
                            this.mainWindow.webContents.toggleDevTools();
                        }
                    }
                ]
            }
        ];

        const menu = Menu.buildFromTemplate(template);
        Menu.setApplicationMenu(menu);
    }

    setupIPC() {
        // Handle backend status requests
        ipcMain.handle('get-backend-status', async () => {
            return {
                isReady: this.isBackendReady,
                url: this.backendUrl,
                port: this.backendPort
            };
        });

        // Handle backend restart
        ipcMain.handle('restart-backend', async () => {
            await this.stopBackend();
            await this.startBackend();
            return this.isBackendReady;
        });

        // Handle app info
        ipcMain.handle('get-app-info', () => {
            return {
                name: 'iLumina',
                version: app.getVersion(),
                platform: process.platform,
                arch: process.arch,
                offline: true
            };
        });
    }

    async startBackend() {
        try {
            console.log('Starting iLumina backend...');
            
            // Path to backend executable
            const backendPath = process.env.NODE_ENV === 'development' 
                ? path.join(__dirname, '..', 'backend', 'app.py')
                : path.join(process.resourcesPath, 'backend.exe');

            // Start backend process
            if (process.env.NODE_ENV === 'development') {
                // Development mode - run Python directly
                this.backendProcess = spawn('python', [backendPath], {
                    cwd: path.join(__dirname, '..', 'backend'),
                    stdio: 'pipe'
                });
            } else {
                // Production mode - run executable
                this.backendProcess = spawn(backendPath, [], {
                    stdio: 'pipe'
                });
            }

            // Handle backend output
            this.backendProcess.stdout.on('data', (data) => {
                console.log(`Backend: ${data}`);
            });

            this.backendProcess.stderr.on('data', (data) => {
                console.error(`Backend Error: ${data}`);
            });

            this.backendProcess.on('close', (code) => {
                console.log(`Backend process exited with code ${code}`);
                this.isBackendReady = false;
            });

            // Wait for backend to be ready
            await this.waitForBackend();

        } catch (error) {
            console.error('Failed to start backend:', error);
            this.isBackendReady = false;
        }
    }

    async waitForBackend(maxRetries = 30) {
        for (let i = 0; i < maxRetries; i++) {
            try {
                const response = await axios.get(`${this.backendUrl}/health`, { timeout: 1000 });
                if (response.status === 200) {
                    this.isBackendReady = true;
                    console.log('Backend is ready!');
                    return;
                }
            } catch (error) {
                // Backend not ready yet, wait and retry
                await new Promise(resolve => setTimeout(resolve, 1000));
            }
        }
        
        console.error('Backend failed to start within timeout period');
        this.isBackendReady = false;
    }

    async stopBackend() {
        if (this.backendProcess) {
            console.log('Stopping backend...');
            this.backendProcess.kill();
            this.backendProcess = null;
            this.isBackendReady = false;
        }
    }

    cleanup() {
        this.stopBackend();
        if (this.mainWindow) {
            this.mainWindow = null;
        }
    }

    async initialize() {
        // Start backend first
        await this.startBackend();
        
        // Create main window
        await this.createWindow();
    }
}

// Create app instance
const iluminaApp = new ILuminaApp();

// App event handlers
app.whenReady().then(() => {
    iluminaApp.initialize();
});

app.on('window-all-closed', () => {
    if (process.platform !== 'darwin') {
        app.quit();
    }
});

app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
        iluminaApp.initialize();
    }
});

app.on('before-quit', () => {
    iluminaApp.cleanup();
});

// Handle app termination
process.on('SIGINT', () => {
    iluminaApp.cleanup();
    process.exit(0);
});

process.on('SIGTERM', () => {
    iluminaApp.cleanup();
    process.exit(0);
});
