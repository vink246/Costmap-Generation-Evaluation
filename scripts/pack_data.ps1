# Packages local raw and processed data into zip archives for sharing.
param(
    [string]$OutDir = "artifacts"
)

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

function Pack-IfExists($Path, $ZipPath) {
    if (Test-Path $Path) {
        Write-Host "Packing $Path -> $ZipPath"
        if (Test-Path $ZipPath) { Remove-Item $ZipPath -Force }
        Compress-Archive -Path $Path -DestinationPath $ZipPath -Force
    } else {
        Write-Warning "Skip: $Path not found"
    }
}

Pack-IfExists -Path "data/raw/kitti" -ZipPath "$OutDir/kitti_raw.zip"
Pack-IfExists -Path "data/raw/nyu_depth_v2" -ZipPath "$OutDir/nyu_raw.zip"
Pack-IfExists -Path "data/processed" -ZipPath "$OutDir/processed.zip"

Write-Host "Done! Upload zips in '$OutDir' to OneDrive/Google Drive/S3 and share read links."
