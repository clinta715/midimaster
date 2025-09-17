Param(
  [string]$ProjectRoot = (Get-Location).Path,
  [string]$ReleaseDir = "releases",
  [string]$StagingName = "midimaster_release_staging"
)

# Fail fast and be strict
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$ProgressPreference = 'SilentlyContinue'

# Paths
$stg = Join-Path $ReleaseDir $StagingName

# Ensure releases directory exists
if (-not (Test-Path $ReleaseDir)) {
  New-Item -ItemType Directory -Path $ReleaseDir | Out-Null
}

# Always recreate staging to avoid stale/locked files
if (Test-Path $stg) {
  try { Remove-Item -Recurse -Force $stg } catch { Start-Sleep -Milliseconds 300; Remove-Item -Recurse -Force $stg }
}

# Build robocopy args: copy project to staging excluding heavy/cache dirs and junk files
$excludeDirs = @(
  ".git","output","reference_midis",".mypy_cache",".pytest_cache",".roo",
  "releases","__pycache__",".benchmarks",".venv",".tox",".ruff_cache",
  ".idea",".vscode",".svn","dist","build"
)
$xd = @()
foreach ($d in $excludeDirs) { $xd += @("/XD", $d) }
$xf = @("/XF", "*.pyc", "*.pyo", ".DS_Store", "Thumbs.db", "*.zip", "*.7z")

# Create staging root
New-Item -ItemType Directory -Path $stg | Out-Null

# Copy with robocopy (quiet logging)
$roboArgs = @("$ProjectRoot", "$stg", "/E", "/R:0", "/W:0", "/NFL", "/NDL", "/NJH", "/NJS", "/NP") + $xd + $xf
# robocopy exits with non-0 for some conditions; ignore exit code and rely on presence of files
& robocopy @roboArgs | Out-Null

# Double-sanitize inside staging in case anything slipped through (and handle name matches anywhere)
$pruneNames = $excludeDirs
Get-ChildItem -Path $stg -Recurse -Force -Directory -ErrorAction SilentlyContinue |
  Where-Object { $pruneNames -contains $_.Name } |
  ForEach-Object {
    try { Remove-Item -Recurse -Force -LiteralPath $_.FullName -ErrorAction SilentlyContinue } catch {}
  }

# Also remove any zip artifacts that might be open/locked in source tree but got copied
Get-ChildItem -Path $stg -Recurse -Force -File -Include *.zip,*.7z -ErrorAction SilentlyContinue |
  ForEach-Object {
    try { Remove-Item -Force -LiteralPath $_.FullName -ErrorAction SilentlyContinue } catch {}
  }

if (-not (Test-Path $stg)) { throw "Staging directory missing: $stg" }

# Create timestamped zip
$ts = Get-Date -Format 'yyyyMMdd_HHmmss'
$zip = Join-Path $ReleaseDir ("midimaster_release_" + $ts + ".zip")
if (Test-Path $zip) { Remove-Item -Force $zip }

# Compress with optimal level
Compress-Archive -Path (Join-Path $stg "*") -DestinationPath $zip -Force -CompressionLevel Optimal

# SHA256 checksum
$hash = (Get-FileHash -Algorithm SHA256 -Path $zip).Hash
$checksumPath = Join-Path $ReleaseDir ("midimaster_release_" + $ts + ".sha256")
"$hash  $(Split-Path -Leaf $zip)" | Out-File -Encoding ascii -FilePath $checksumPath

# Clean up staging
try { Remove-Item -Recurse -Force $stg } catch { Start-Sleep -Milliseconds 300; Remove-Item -Recurse -Force $stg }

# Output summary
Write-Host ("ZIP_PATH=" + $zip)
Write-Host ("SHA256=" + $hash)
Write-Host ("CHECKSUM_PATH=" + $checksumPath)