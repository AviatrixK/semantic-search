<#
Windows stand-in for a Makefile. Dot-source it, then call the functions:
    . .\scripts\dev.ps1
    Up; Logs api; Admin me@example.com 'StrongPass123'; Test; Down
#>
$Script:Root = Split-Path -Parent $PSScriptRoot

function Up {
    if (-not (Test-Path (Join-Path $Script:Root '.env'))) {
        Copy-Item (Join-Path $Script:Root '.env.example') (Join-Path $Script:Root '.env')
        Write-Warning 'Created .env from .env.example - set JWT_SECRET before using this for real.'
    }
    docker compose --project-directory $Script:Root up -d --build
}

function Down {
    docker compose --project-directory $Script:Root down
}

function Logs {
    param([string]$Service = '')
    if ($Service) { docker compose --project-directory $Script:Root logs -f --tail=100 $Service }
    else { docker compose --project-directory $Script:Root logs -f --tail=100 }
}

function Admin {
    param(
        [Parameter(Mandatory)][string]$Email,
        [Parameter(Mandatory)][string]$Password
    )
    docker compose --project-directory $Script:Root exec api python -m app.scripts.create_admin $Email $Password
}

function Test {
    $py = Join-Path $Script:Root 'backend\.venv\Scripts\python.exe'
    if (-not (Test-Path $py)) { $py = 'python' }
    Push-Location (Join-Path $Script:Root 'backend')
    try { & $py -m pytest -q @args } finally { Pop-Location }
}
