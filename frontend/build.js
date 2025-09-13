/**
 * Build script for iLumina Frontend
 * Builds Electron app and creates installer
 */

const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');

function runCommand(command, cwd = process.cwd()) {
    console.log(`Running: ${command}`);
    try {
        execSync(command, { 
            cwd, 
            stdio: 'inherit',
            shell: true 
        });
        return true;
    } catch (error) {
        console.error(`Command failed: ${command}`);
        console.error(error.message);
        return false;
    }
}

function checkPrerequisites() {
    console.log('Checking prerequisites...');
    
    // Check if Node.js is installed
    try {
        const nodeVersion = execSync('node --version', { encoding: 'utf8' }).trim();
        console.log(`Node.js version: ${nodeVersion}`);
    } catch (error) {
        console.error('Node.js is not installed or not in PATH');
        return false;
    }
    
    // Check if npm is installed
    try {
        const npmVersion = execSync('npm --version', { encoding: 'utf8' }).trim();
        console.log(`npm version: ${npmVersion}`);
    } catch (error) {
        console.error('npm is not installed or not in PATH');
        return false;
    }
    
    return true;
}

function installDependencies() {
    console.log('Installing dependencies...');
    return runCommand('npm install');
}

function buildElectronApp() {
    console.log('Building Electron app...');
    
    // First, ensure backend is built
    const backendPath = path.join(__dirname, '..', 'backend');
    if (!fs.existsSync(path.join(backendPath, 'dist', 'backend.exe'))) {
        console.log('Backend not found. Building backend first...');
        if (!runCommand('python build.py', backendPath)) {
            console.error('Failed to build backend');
            return false;
        }
    }
    
    // Copy backend executable to resources
    const backendExe = path.join(backendPath, 'dist', 'backend.exe');
    const resourcesDir = path.join(__dirname, 'resources');
    
    if (!fs.existsSync(resourcesDir)) {
        fs.mkdirSync(resourcesDir, { recursive: true });
    }
    
    if (fs.existsSync(backendExe)) {
        fs.copyFileSync(backendExe, path.join(resourcesDir, 'backend.exe'));
        console.log('Backend executable copied to resources');
    } else {
        console.error('Backend executable not found');
        return false;
    }
    
    // Build Electron app
    return runCommand('npm run build');
}

function createInstaller() {
    console.log('Creating installer...');
    return runCommand('npm run dist');
}

function main() {
    console.log('iLumina Frontend Build Script');
    console.log('=' * 40);
    
    // Check prerequisites
    if (!checkPrerequisites()) {
        console.error('Prerequisites check failed');
        process.exit(1);
    }
    
    // Install dependencies
    if (!installDependencies()) {
        console.error('Failed to install dependencies');
        process.exit(1);
    }
    
    // Build Electron app
    if (!buildElectronApp()) {
        console.error('Failed to build Electron app');
        process.exit(1);
    }
    
    // Create installer
    if (!createInstaller()) {
        console.error('Failed to create installer');
        process.exit(1);
    }
    
    console.log('\n✅ Frontend build completed successfully!');
    console.log('\nInstaller created in frontend/dist/ directory');
    console.log('You can now distribute the iLumina-Setup.exe installer');
}

if (require.main === module) {
    main();
}

module.exports = {
    checkPrerequisites,
    installDependencies,
    buildElectronApp,
    createInstaller
};
