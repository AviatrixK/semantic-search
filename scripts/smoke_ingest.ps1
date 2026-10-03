<#
End-to-end ingestion smoke test against a running stack (docker compose up).

  .\scripts\smoke_ingest.ps1 -Email admin@example.com -Password 'ChangeMe123!'
  .\scripts\smoke_ingest.ps1 -Email ... -Password ... -File "sample\other.mp4" -ExpectedChunks 59
  .\scripts\smoke_ingest.ps1 -Email ... -Password ... -File "sample\EduSphereDemonstration.mp4" -ExpectNoSpeech

Steps: login -> upload -> verify storage by GETting the pre-signed stream URL -> poll the job -> print chunk count.
Storage is SeaweedFS (no console), so the pre-signed fetch is the storage check.
Exit code 0 = behaved as expected, 1 = anything else.
#>
param(
    [string]$Email = $env:SVS_EMAIL,
    [string]$Password = $env:SVS_PASSWORD,
    [string]$File = "sample\purpose.mp4",
    [string]$ApiBase = "http://localhost:8000",
    [int]$ExpectedChunks = 0,        # e.g. 59 for sample\purpose.mp4; 0 = just report
    [switch]$ExpectNoSpeech,         # pass for the silent sample: success means failed with "No speech detected"
    [int]$TimeoutMin = 60
)
$ErrorActionPreference = "Stop"

function Fail([string]$msg) { Write-Host "FAIL: $msg" -ForegroundColor Red; exit 1 }

if (-not $Email -or -not $Password) { Fail "Pass -Email and -Password (or set SVS_EMAIL / SVS_PASSWORD)." }
$root = Split-Path -Parent $PSScriptRoot
$path = if ([System.IO.Path]::IsPathRooted($File)) { $File } else { Join-Path $root $File }
if (-not (Test-Path -LiteralPath $path)) { Fail "File not found: $path" }
if (-not (Get-Command curl.exe -ErrorAction SilentlyContinue)) { Fail "curl.exe not found (ships with Windows 10+)." }

# 1. login
$login = Invoke-RestMethod -Method Post -Uri "$ApiBase/auth/login" -ContentType "application/json" `
    -Body (@{ email = $Email; password = $Password } | ConvertTo-Json)
$auth = @{ Authorization = "Bearer $($login.access_token)" }
Write-Host "Logged in as $Email"

# 2. upload (curl.exe: Windows PowerShell 5.1 has no multipart -Form)
$item = Get-Item -LiteralPath $path
Write-Host ("Uploading {0} ({1:N1} MB)..." -f $item.Name, ($item.Length / 1MB))
$raw = & curl.exe -s -w "`n%{http_code}" -X POST "$ApiBase/api/videos" -H "Authorization: Bearer $($login.access_token)" `
    -F "file=@$($item.FullName);type=video/mp4"
$lines = @($raw)
$code = $lines[-1]
$body = ($lines[0..($lines.Count - 2)] -join "`n")
if ($code -ne "202") { Fail "Upload returned HTTP $code : $body" }
$upload = $body | ConvertFrom-Json
Write-Host "video_id=$($upload.video_id) job_id=$($upload.job_id)"

# 3. storage check: fetch the first KB through the pre-signed URL
$stream = Invoke-RestMethod -Uri "$ApiBase/api/videos/$($upload.video_id)/stream" -Headers $auth
try {
    $head = Invoke-WebRequest -UseBasicParsing -Uri $stream.url -Headers @{ Range = "bytes=0-1023" }
} catch {
    Fail "Pre-signed URL did not return the file ($($stream.url)): $($_.Exception.Message)"
}
if ($head.StatusCode -notin 200, 206 -or $head.RawContentLength -le 0) { Fail "Pre-signed fetch returned HTTP $($head.StatusCode), $($head.RawContentLength) bytes" }
Write-Host "Storage OK: pre-signed URL served $($head.RawContentLength) bytes (HTTP $($head.StatusCode))"

# 4. poll the job
$deadline = (Get-Date).AddMinutes($TimeoutMin)
$last = ""
do {
    Start-Sleep -Seconds 2
    $job = Invoke-RestMethod -Uri "$ApiBase/api/jobs/$($upload.job_id)" -Headers $auth
    $line = "stage=$($job.stage) progress=$($job.progress)"
    if ($line -ne $last) { Write-Host $line; $last = $line }
    if ((Get-Date) -gt $deadline) { Fail "Timed out after $TimeoutMin min at $line" }
} while ($job.stage -notin "done", "failed")

# 5. verdict
if ($job.stage -eq "failed") {
    Write-Host "Job failed: $($job.error)"
    if ($ExpectNoSpeech -and $job.error -eq "No speech detected") {
        Write-Host "PASS: silent video failed with the expected message." -ForegroundColor Green
        exit 0
    }
    Fail "Job failed unexpectedly."
}
if ($ExpectNoSpeech) { Fail "Expected 'No speech detected' but the job finished with stage=done." }

$chunks = @(Invoke-RestMethod -Uri "$ApiBase/api/videos/$($upload.video_id)/transcript" -Headers $auth)
if ($chunks.Count -eq 0) { Fail "Job is done but there are 0 chunks." }
$first = $chunks[0].start_sec
$lastEnd = $chunks[-1].end_sec
Write-Host ("chunks: {0}  (covering {1:N1}s - {2:N1}s)" -f $chunks.Count, $first, $lastEnd) -ForegroundColor Green
if ($ExpectedChunks -gt 0) {
    if ($chunks.Count -eq $ExpectedChunks) { Write-Host "Matches expected chunk count ($ExpectedChunks)." -ForegroundColor Green }
    else { Write-Host "Expected $ExpectedChunks chunks, got $($chunks.Count). Compare WHISPER_SIZE and CHUNK_SECONDS/CHUNK_OVERLAP with your local run." -ForegroundColor Yellow }
}
Write-Host "PASS: ingestion finished (stage=done)." -ForegroundColor Green
exit 0
