[CmdletBinding()]
param(
    [DateTimeOffset]$NotBefore = ([DateTimeOffset]::Now.Date.AddHours(4)),
    [string]$Repository = ''
)

$ErrorActionPreference = 'Stop'
if ([string]::IsNullOrWhiteSpace($Repository)) {
    $Repository = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
}

Write-Output "WAITING until $($NotBefore.ToString('o')); repository: $Repository"
while ([DateTimeOffset]::Now -lt $NotBefore) {
    $remaining = ($NotBefore - [DateTimeOffset]::Now).TotalSeconds
    Start-Sleep -Milliseconds ([int][Math]::Ceiling([Math]::Min(60000, [Math]::Max(1, $remaining * 1000))))
}

Write-Output "Pulling origin/main at $([DateTimeOffset]::Now.ToString('o'))"
# Fast-forward only: leave local changes and divergent history untouched.
& git -C $Repository pull --ff-only origin main
if ($LASTEXITCODE -ne 0) {
    throw "Source pull failed (exit $LASTEXITCODE); no reset, clean, or automatic stash was attempted."
}
Write-Output 'SOURCE_READY'
