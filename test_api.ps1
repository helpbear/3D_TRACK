param(
    [string]$ServerUrl = "http://127.0.0.1:8000",
    [string]$TaskId = "activity-recognition",
    [string]$Procedure = "endoscopic-submucosal-dissection",
    [string]$Note = "test frame",
    [string]$ImagePath = "E:\PythonProject\3D_TRACK\EndoARSS\semi-synthetic\frame.png"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path -LiteralPath $ImagePath)) {
    Write-Error "Image not found: $ImagePath"
    exit 1
}

$healthUrl = "$ServerUrl/api/v1/health"
$analyzeUrl = "$ServerUrl/api/v1/assistant/analyze-frame"

try {
    $null = Invoke-WebRequest -Uri $healthUrl -Method Get -TimeoutSec 5 -UseBasicParsing | Out-Null
} catch {
    Write-Error "Backend is not reachable at $ServerUrl. Start it with: python main.py"
    exit 1
}

$curlArgs = @(
    "-sS",
    "-X", "POST",
    $analyzeUrl,
    "-F", "task_id=$TaskId",
    "-F", "procedure=$Procedure",
    "-F", "note=$Note",
    "-F", "frame=@$ImagePath"
)

Write-Host "POST $analyzeUrl"
Write-Host "Image: $ImagePath"
Write-Host ""

& curl.exe @curlArgs
