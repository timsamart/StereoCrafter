<#
.SYNOPSIS
    Batch processes video files through StereoCrafter's depth splatting and stereo inpainting stages.

.DESCRIPTION
    This script takes a batch of video file paths (or processes all .mp4 files in a directory) and runs the two inference steps:
    1. Depth-based video splatting using DepthCrafter.
    2. Stereo video inpainting using StereoCrafter (with a tile number of 2).
    
    The output files are saved in the specified output directory.
    
.PARAMETER VideoFiles
    An array of paths to input video files. Alternatively, use the -InputDir parameter to process all .mp4 files in a directory.
    
.PARAMETER OutputDir
    The directory where processed videos will be saved. Default is ".\outputs".

.PARAMETER InputDir
    (Optional) A directory from which all .mp4 files will be processed. If provided, this overrides the VideoFiles parameter.

.EXAMPLE
    # Process specific video files:
    .\ProcessVideos.ps1 -VideoFiles "C:\videos\video1.mp4", "C:\videos\video2.mp4" -OutputDir ".\outputs"

.EXAMPLE
    # Process all .mp4 files in a directory:
    .\ProcessVideos.ps1 -InputDir "C:\videos" -OutputDir ".\outputs"
#>

param(
    [Parameter(ParameterSetName = "Files", Mandatory = $false)]
    [string[]]$VideoFiles,

    [Parameter(ParameterSetName = "Dir", Mandatory = $false)]
    [string]$InputDir,

    [Parameter(Mandatory = $false)]
    [string]$OutputDir = ".\outputs"
)

# If InputDir is provided, override VideoFiles with all .mp4 files in that directory.
if ($InputDir) {
    if (-not (Test-Path $InputDir)) {
        Write-Error "Input directory '$InputDir' does not exist."
        exit 1
    }
    $VideoFiles = Get-ChildItem -Path $InputDir -Filter *.mp4 | ForEach-Object { $_.FullName }
    if ($VideoFiles.Count -eq 0) {
        Write-Error "No .mp4 files found in directory '$InputDir'."
        exit 1
    }
}

if (-not $VideoFiles -or $VideoFiles.Count -eq 0) {
    Write-Error "No video files provided. Use -VideoFiles or -InputDir to specify videos."
    exit 1
}

# Ensure output directory exists
if (-not (Test-Path $OutputDir)) {
    Write-Output "Creating output directory '$OutputDir'..."
    New-Item -ItemType Directory -Path $OutputDir | Out-Null
}

# Set weights and model paths (adjust these if necessary)
$SVDModelPath    = ".\weights\stable-video-diffusion-img2vid-xt-1-1"
$DepthModelPath  = ".\weights\DepthCrafter"
$StereoModelPath = ".\weights\StereoCrafter"

# Process each video file
foreach ($video in $VideoFiles) {
    $baseName = [System.IO.Path]::GetFileNameWithoutExtension($video)
    $splatOutput = Join-Path $OutputDir "$baseName`_splat.mp4"
    Write-Output "---------------------------------------------------"
    Write-Output "Processing '$video'..."
    
    # STEP 1: Depth-based Video Splatting
    $depthCmd = "python depth_splatting_inference.py --pre_trained_path `"$SVDModelPath`" --unet_path `"$DepthModelPath`" --input_video_path `"$video`" --output_video_path `"$splatOutput`""
    Write-Output "Running depth splatting:"
    Write-Output $depthCmd
    try {
        Invoke-Expression $depthCmd
    }
    catch {
        Write-Error "Depth splatting failed for '$video'. Skipping to next file."
        continue
    }
    
    # STEP 2: Stereo Video Inpainting with --tile_num 2 (as in your .sh script)
    $inpaintCmd = "python inpainting_inference.py --pre_trained_path `"$SVDModelPath`" --unet_path `"$StereoModelPath`" --input_video_path `"$splatOutput`" --save_dir `"$OutputDir`" --tile_num 2"
    Write-Output "Running stereo inpainting:"
    Write-Output $inpaintCmd
    try {
        Invoke-Expression $inpaintCmd
    }
    catch {
        Write-Error "Stereo inpainting failed for '$video'."
        continue
    }
    
    Write-Output "Completed processing '$video'."
}

Write-Output "---------------------------------------------------"
Write-Output "All videos processed. Time to sit back and enjoy some 3D magic!"
