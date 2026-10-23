# Deploy the hub to Cloud Run. Defaults chosen to sit inside the always-free tier:
# scale to zero between visits, one small instance, a low ceiling.
#
#   .\scripts\deploy.ps1
#   .\scripts\deploy.ps1 -Region europe-west1 -Service convo-portfolio

param(
    [string]$Service = "convo-portfolio",
    [string]$Region = "us-central1",
    [string]$Project = "",
    [switch]$SkipApis
)

$ErrorActionPreference = "Stop"

if (-not (Get-Command gcloud -ErrorAction SilentlyContinue)) {
    Write-Error @'
gcloud is not on PATH. Install the Google Cloud CLI first:
  https://cloud.google.com/sdk/docs/install
Then: gcloud init  (and gcloud auth login)
'@
}

if ($Project -eq "") {
    $Project = (gcloud config get-value project 2>$null)
    if ($?) { $Project = $Project.Trim() }
}
if ([string]::IsNullOrWhiteSpace($Project) -or $Project -eq "(unset)") {
    Write-Error "No project set. Pass -Project <id>, or run: gcloud config set project <id>"
}

Write-Host "Deploying '$Service' to $Region in project $Project" -ForegroundColor Cyan

if (-not $SkipApis) {
    Write-Host "Enabling the three APIs this needs (idempotent)..." -ForegroundColor DarkGray
    gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com --project $Project
    if (-not $?) { Write-Error "Could not enable APIs. Is billing enabled on $Project?" }
}

# --source builds the Dockerfile with Cloud Build, so no local docker push is needed.
gcloud run deploy $Service `
    --source . `
    --project $Project `
    --region $Region `
    --allow-unauthenticated `
    --port 8080 `
    --cpu 1 `
    --memory 1Gi `
    --min-instances 0 `
    --max-instances 3 `
    --concurrency 20 `
    --timeout 300

if (-not $?) { Write-Error "Deploy failed. Check the Cloud Build log URL above." }

$url = (gcloud run services describe $Service --project $Project --region $Region --format "value(status.url)")
Write-Host ""
Write-Host "Live at: $url" -ForegroundColor Green
Write-Host "min-instances is 0, so the first request after a quiet spell takes a few seconds." -ForegroundColor DarkGray
