# Verify seki-inference-engine on localhost:9000 (compose host publish).
$ErrorActionPreference = "Stop"
$base = $env:INFERENCE_URL
if (-not $base) { $base = "http://localhost:9000" }
$base = $base.TrimEnd("/")
if ($base.EndsWith("/v1")) { $base = $base.Substring(0, $base.Length - 3).TrimEnd("/") }
$key = $env:INFERENCE_API_KEY
if (-not $key) { $key = $env:API_KEY }
if (-not $key) { $key = "change-me" }

Write-Host "== GET $base/health =="
$health = Invoke-RestMethod -Uri "$base/health"
$health | ConvertTo-Json -Compress
if ($health.status -ne "ok") { throw "health failed" }

Write-Host "`n== GET $base/ready =="
$ready = $null
for ($i = 0; $i -lt 15; $i++) {
  try {
    $ready = Invoke-WebRequest -Uri "$base/ready" -UseBasicParsing
    if ($ready.StatusCode -eq 200) { break }
  } catch {
    $resp = $_.Exception.Response
    if ($resp) {
      $code = [int]$resp.StatusCode
      Write-Host "HTTP $code (retry $($i + 1)/15)"
    }
    Start-Sleep -Seconds 2
  }
}
if (-not $ready -or $ready.StatusCode -ne 200) {
  throw "ready is not 200 (is Ollama up? try: Invoke-WebRequest http://localhost:9000/ready)"
}
Write-Host "HTTP $($ready.StatusCode)  <-- 200 means Ollama /v1/models answered, not that the GGUF is in VRAM"
Write-Host $ready.Content

Write-Host "`n== POST $base/v1/chat/completions (first load can take several minutes) =="
$headers = @{
  Authorization = "Bearer $key"
  "Content-Type" = "application/json"
}
$body = '{"messages":[{"role":"user","content":"Say hi in one word."}],"max_tokens":16}'
try {
  $chat = Invoke-WebRequest -Uri "$base/v1/chat/completions" -Method POST -Headers $headers -Body $body -TimeoutSec 360 -UseBasicParsing
} catch {
  $resp = $_.Exception.Response
  if ($resp) {
    $errBody = [System.IO.StreamReader]::new($resp.GetResponseStream()).ReadToEnd()
    Write-Host "HTTP $([int]$resp.StatusCode)"
    Write-Host $errBody
    throw "chat failed: first CUDA load often exceeds a 120s gateway timeout. Raise REQUEST_TIMEOUT and recreate the inference container."
  }
  throw
}
$backend = $chat.Headers["x-seki-backend"]
Write-Host "HTTP $($chat.StatusCode)"
Write-Host "x-seki-backend: $backend"
Write-Host $chat.Content
if ($chat.StatusCode -ne 200) { throw "chat failed" }
if (-not $backend) { throw "missing x-seki-backend header" }
Write-Host "`nOK - completion succeeded via $backend"
