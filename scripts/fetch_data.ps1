# Downloads and extracts shared archives to data/raw and data/processed.
# Usage:
#   .\scripts\fetch_data.ps1 -KittiZipUrl "https://.../kitti_raw.zip" -NyuZipUrl "https://.../nyu_raw.zip" -ProcessedZipUrl "https://.../processed.zip"

param(
    [string]$KittiZipUrl = "",
    [string]$NyuZipUrl = "",
    [string]$ProcessedZipUrl = "",
    [string]$TmpDir = "downloads"
)

New-Item -ItemType Directory -Force -Path $TmpDir | Out-Null

function Get-IfUrl($Url, $OutFile) {
    if ($Url -and $Url.Trim() -ne "") {
        Write-Host "Downloading $Url -> $OutFile"
        Invoke-WebRequest -Uri $Url -OutFile $OutFile -UseBasicParsing
    } else {
        Write-Host "Skip download: no URL provided for $OutFile"
    }
}

function Extract-IfExists($ZipPath, $Dest) {
    if (Test-Path $ZipPath) {
        Write-Host "Extracting $ZipPath -> $Dest"
        New-Item -ItemType Directory -Force -Path $Dest | Out-Null
        Expand-Archive -Path $ZipPath -DestinationPath $Dest -Force
    } else {
        Write-Host "Skip extract: $ZipPath not found"
    }
}

$kittiZip = Join-Path $TmpDir "kitti_raw.zip"
$nyuZip = Join-Path $TmpDir "nyu_raw.zip"
$procZip = Join-Path $TmpDir "processed.zip"

Get-IfUrl -Url $KittiZipUrl -OutFile $kittiZip
Get-IfUrl -Url $NyuZipUrl -OutFile $nyuZip
Get-IfUrl -Url $ProcessedZipUrl -OutFile $procZip

Extract-IfExists -ZipPath $kittiZip -Dest "data/raw"
Extract-IfExists -ZipPath $nyuZip -Dest "data/raw"
Extract-IfExists -ZipPath $procZip -Dest "data"

Write-Host "Done! Verify directories: data/raw/kitti, data/raw/nyu_depth_v2, data/processed."
