# InitializeStereoCrafter.ps1
# This script automates the setup of the StereoCrafter project.
# It assumes you have Git, Python 3.8+ and Git LFS installed.
# Note: Building the Forward-Warp dependency requires Bash (e.g., Git Bash or WSL).

# Function to check if a command exists
function Command-Exists {
    param([string]$command)
    return (Get-Command $command -ErrorAction SilentlyContinue) -ne $null
}

# Check for Git
if (-not (Command-Exists "git")) {
    Write-Error "Git is not installed. Please install Git before running this script."
    exit 1
}

# Check for Python
if (-not (Command-Exists "python")) {
    Write-Error "Python is not installed. Please install Python 3.8 or later before running this script."
    exit 1
}

# Clone the StereoCrafter repository (with submodules) if not already cloned
if (-not (Test-Path "./StereoCrafter")) {
    Write-Output "Cloning the StereoCrafter repository... (this is where the magic begins)"
    git clone --recursive https://github.com/TencentARC/StereoCrafter.git
} else {
    Write-Output "StereoCrafter repository already exists. Skipping clone."
}

# Enter the project directory
Set-Location "./StereoCrafter"

# Install Python dependencies
Write-Output "Installing Python dependencies... Sit back and let pip do the work."
python -m pip install -r requirements.txt

# Build the Forward-Warp dependency
if (Test-Path "./dependency/Forward-Warp") {
    Write-Output "Building Forward-Warp dependency... Time to forward your warp drive!"
    Set-Location "./dependency/Forward-Warp"
    
    # Run the install script using bash (required for executing .sh on Windows)
    if (Command-Exists "bash") {
        bash -c "chmod a+x install.sh && ./install.sh"
    } else {
        Write-Error "Bash is not installed. Please install Git Bash or WSL to build the Forward-Warp dependency."
        exit 1
    }
    # Return to project root
    Set-Location "../../"
} else {
    Write-Warning "Forward-Warp dependency directory not found. Skipping its build step."
}

# Create and enter the weights directory
if (-not (Test-Path "./weights")) {
    Write-Output "Creating weights directory..."
    New-Item -ItemType Directory -Path "./weights" | Out-Null
}
Set-Location "./weights"

# Initialize Git LFS for handling large files
Write-Output "Initializing Git LFS... Because big models need big handling."
git lfs install

# Clone model weights if they don't already exist

# 1. SVD img2vid model weights
if (-not (Test-Path "./stable-video-diffusion-img2vid-xt-1-1")) {
    Write-Output "Cloning SVD img2vid model weights..."
    git clone https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt-1-1
} else {
    Write-Output "SVD img2vid model weights already exist. Skipping clone."
}

# 2. DepthCrafter model weights
if (-not (Test-Path "./DepthCrafter")) {
    Write-Output "Cloning DepthCrafter model weights..."
    git clone https://huggingface.co/tencent/DepthCrafter
} else {
    Write-Output "DepthCrafter model weights already exist. Skipping clone."
}

# 3. StereoCrafter model weights
if (-not (Test-Path "./StereoCrafter")) {
    Write-Output "Cloning StereoCrafter model weights..."
    git clone https://huggingface.co/TencentARC/StereoCrafter
} else {
    Write-Output "StereoCrafter model weights already exist. Skipping clone."
}

Write-Output "Project initialization complete! You’re now set to generate some immersive stereoscopic 3D videos."

