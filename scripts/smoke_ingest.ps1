<#
End-to-end ingestion smoke test against a running stack (docker compose up). Works on Windows PowerShell 5.1.

  .\scripts\smoke_ingest.ps1 -Email admin@example.com -Password 'ChangeMe123!'
  .\scripts\smoke_ingest.ps1 -Email ... -Password ... -File "sample\purpose.mp4" -ExpectedChunks 59
  .\scripts\smoke_ingest.ps1 -Email ... -Password ... -File "sample\EduSphereDemonstration.mp4" -ExpectNoSpeech
  .\scripts\smoke_ingest.ps1 -Email ... -Password ... -File "sample\fake.mp4" -ExpectFail     # e.g. a renamed .txt

Steps: login -> upload -> check storage by fetching the pre-signed stream URL (non-fatal) -> poll the job -> chunk count.
Storage is SeaweedFS (no console), so the pre-signed fetch is the storage check.
  -ExpectFail      pass when the job ends "failed" (prints the error message); fail if it finishes "done".
  -ExpectNoSpeech  like -ExpectFail, but the error must be exactly "No speech detected in this video."
Exit code 0 = behaved as expected, 1 = anything else. Any non-2xx HTTP response prints its body.
#>
param(
    [string]$Email = $env:SVS_EMAIL,
    [string]$Password = $env:SVS_PASSWORD,
    [string]$File = "sample\purpose.mp4",
    [string]$ApiBase = "http://localhost:8000",
    [int]$ExpectedChunks = 0,        # e.g. 59 for sample\purpose.mp4; 0 = just report
    [switch]$ExpectFail,
    [switch]$ExpectNoSpeech,
    [int]$TimeoutMin = 60
)
$ErrorActionPreference = "Stop"
$NoSpeechMessage = "No speech detected in this video."

function Fail([string]$msg) { Write-Host "FAIL: $msg" -ForegroundColor Red; exit 1 }

# Calls the API, returns parsed JSON. On any non-2xx (or network error) prints the response body and exits 1.
function Invoke-Api {
    param([string]$Method, [string]$Uri, [hashtable]$Headers = @{}, [string]$Body = $null)
    try {
        $args2 = @{ UseBasicParsing = $true; Method = $Method; Uri = $Uri; Headers = $Headers }
        if ($Body) { $args2.Body = $Body; $args2.ContentType = "application/json" }
        $r = Invoke-WebRequest @args2
        if ([string]::IsNullOrWhiteSpace($r.Content)) { return $null }
        return ($r.Content | ConvertFrom-Json)
    } catch {
        $status = "n/a"; $text = $_.Exception.Message
        $resp = $_.Exception.Response
        if ($resp) {
            $status = [int]$resp.StatusCode
            if ($_.ErrorDetails -and $_.ErrorDetails.Message) { $text = $_.ErrorDetails.Message }
            else {
                try { $text = (New-Object System.IO.StreamReader($resp.GetResponseStream())).ReadToEnd() } catch { }
            }
        }
        Fail "$Method $Uri -> HTTP $status`n$text"
    }
}

if (-not $Email -or -not $Password) { Fail "Pass -Email and -Password (or set SVS_EMAIL / SVS_PASSWORD)." }
$root = Split-Path -Parent $PSScriptRoot
$path = if ([System.IO.Path]::IsPathRooted($File)) { $File } else { Join-Path $root $File }
if (-not (Test-Path -LiteralPath $path)) { Fail "File not found: $path" }
if (-not (Get-Command curl.exe -ErrorAction SilentlyContinue)) { Fail "curl.exe not found (ships with Windows 10+)." }

# 1. login
$login = Invoke-Api -Method Post -Uri "$ApiBase/auth/login" -Body (@{ email = $Email; password = $Password } | ConvertTo-Json)
$auth = @{ Authorization = "Bearer $($login.access_token)" }
Write-Host "Logged in as $Email"

# 2. upload (curl.exe: Windows PowerShell 5.1 has no multipart -Form)
$item = Get-Item -LiteralPath $path
$mime = switch ($item.Extension.ToLower()) {
    ".webm" { "video/webm" } ".mov" { "video/quicktime" } ".mkv" { "video/x-matroska" } default { "video/mp4" }
}
Write-Host ("Uploading {0} ({1:N1} MB)..." -f $item.Name, ($item.Length / 1MB))
$tmp = [System.IO.Path]::GetTempFileName()
try {
    $code = (& curl.exe -s -o $tmp -w "%{http_code}" -X POST "$ApiBase/api/videos" `
        -H "Authorization: Bearer $($login.access_token)" -F "file=@$($item.FullName);type=$mime") | Out-String
    $code = $code.Trim()
    $body = Get-Content -LiteralPath $tmp -Raw
} finally { Remove-Item -LiteralPath $tmp -ErrorAction SilentlyContinue }
if ($code -notmatch "^2\d\d$") { Fail "Upload -> HTTP $code`n$body" }
$upload = $body | ConvertFrom-Json
Write-Host "video_id=$($upload.video_id) job_id=$($upload.job_id)"

# 3. storage check (non-fatal): first KB through the pre-signed URL. curl.exe because Windows PowerShell 5.1
#    refuses to set the Range header via Invoke-WebRequest.
$stream = Invoke-Api -Method Get -Uri "$ApiBase/api/videos/$($upload.video_id)/stream" -Headers $auth
$url = $stream.url
$tmp = [System.IO.Path]::GetTempFileName()
try {
    $code = (& curl.exe -s -o $tmp -w "%{http_code}" -r 0-1023 "$url") | Out-String
    $code = $code.Trim()
    if ($code -eq "200" -or $code -eq "206") {
        Write-Host "Storage OK: pre-signed URL served the file (HTTP $code)"
    } else {
        $text = ""
        if (Test-Path -LiteralPath $tmp) { $text = Get-Content -LiteralPath $tmp -Raw -ErrorAction SilentlyContinue }
        if ($text -and $text.Length -gt 500) { $text = $text.Substring(0, 500) + "..." }
        Write-Warning "Pre-signed URL check returned HTTP $code for $url`n$text"
        Write-Warning "Continuing to job polling."
    }
} finally { Remove-Item -LiteralPath $tmp -ErrorAction SilentlyContinue }

# 4. poll the job
$deadline = (Get-Date).AddMinutes($TimeoutMin)
$last = ""
do {
    Start-Sleep -Seconds 2
    $job = Invoke-Api -Method Get -Uri "$ApiBase/api/jobs/$($upload.job_id)" -Headers $auth
    $line = "stage=$($job.stage) progress=$($job.progress)"
    if ($job.error) { $line += " error=$($job.error)" }
    if ($line -ne $last) { Write-Host $line; $last = $line }
    if ((Get-Date) -gt $deadline) { Fail "Timed out after $TimeoutMin min at $line" }
} while ($job.stage -ne "done" -and $job.stage -ne "failed")

# 5. verdict
if ($job.stage -eq "failed") {
    Write-Host "Job failed: $($job.error)"
    if ($job.error -match '(/tmp|[A-Za-z]:\\|0x[0-9a-fA-F]{4,}|ffprobe|ffmpeg|Traceback)') {
        Write-Warning "jobs.error looks like it contains internals (paths/tool output). It should be a short user-facing message."
    }
    if ($ExpectNoSpeech) {
        if ($job.error -eq $NoSpeechMessage) {
            Write-Host "PASS: silent video failed with the expected message." -ForegroundColor Green; exit 0
        }
        Fail "Expected error '$NoSpeechMessage' but got '$($job.error)'."
    }
    if ($ExpectFail) { Write-Host "PASS: job failed as expected." -ForegroundColor Green; exit 0 }
    Fail "Job failed unexpectedly."
}
if ($ExpectFail -or $ExpectNoSpeech) { Fail "Expected the job to fail but it finished with stage=done." }

$chunks = @(Invoke-Api -Method Get -Uri "$ApiBase/api/videos/$($upload.video_id)/transcript" -Headers $auth)
if ($chunks.Count -eq 0) { Fail "Job is done but there are 0 chunks." }
Write-Host ("chunks: {0}  (covering {1:N1}s - {2:N1}s)" -f $chunks.Count, $chunks[0].start_sec, $chunks[-1].end_sec) -ForegroundColor Green
if ($ExpectedChunks -gt 0) {
    if ($chunks.Count -eq $ExpectedChunks) { Write-Host "Matches expected chunk count ($ExpectedChunks)." -ForegroundColor Green }
    else { Write-Host "Expected $ExpectedChunks chunks, got $($chunks.Count). Compare WHISPER_SIZE and CHUNK_SECONDS/CHUNK_OVERLAP with your local run." -ForegroundColor Yellow }
}
Write-Host "PASS: ingestion finished (stage=done)." -ForegroundColor Green
exit 0
