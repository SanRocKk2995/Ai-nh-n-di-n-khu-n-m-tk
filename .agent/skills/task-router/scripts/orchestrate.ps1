param(
    [string]$ImplementPrompt = "prompts/implement.txt",
    [string]$ReviewPrompt = "prompts/review.txt"
)

$logDir = "logs"
$orchLog = "$logDir/orchestrate.out"
$resultJson = "$logDir/result.json"

New-Item -ItemType Directory -Force -Path $logDir | Out-Null

"Workflow started" | Tee-Object $orchLog
"Implement prompt: $ImplementPrompt" | Tee-Object -Append $orchLog
"Review prompt: $ReviewPrompt" | Tee-Object -Append $orchLog
"--------------------------------------" | Tee-Object -Append $orchLog

$codexStatus = "not_run"
$copilotStatus = "not_run"

# Step 1: Codex
try {
    powershell -ExecutionPolicy Bypass -File ".agent/skills/task-router/scripts/run_codex.ps1" $ImplementPrompt
    $codexStatus = "ok"
}
catch {
    $codexStatus = "failed"
}

# Step 2: Copilot
try {
    powershell -ExecutionPolicy Bypass -File ".agent/skills/task-router/scripts/run_copilot.ps1" $ReviewPrompt
    $copilotStatus = "ok"
}
catch {
    $copilotStatus = "failed"
}

$overall = "ok"
if ($codexStatus -ne "ok" -or $copilotStatus -ne "ok") {
    $overall = "partial_or_failed"
}

$result = @{
    status = $overall
    workflow = "antigravity -> codex -> copilot"
    steps = @(
        @{
            name = "implement"
            agent = "codex"
            status = $codexStatus
            log = "logs/codex.out"
        },
        @{
            name = "review"
            agent = "copilot"
            status = $copilotStatus
            log = "logs/copilot.out"
        }
    )
}

$result | ConvertTo-Json -Depth 4 | Set-Content $resultJson

"--------------------------------------" | Tee-Object -Append $orchLog
"Result written to $resultJson" | Tee-Object -Append $orchLog
"Workflow finished: $overall" | Tee-Object -Append $orchLog