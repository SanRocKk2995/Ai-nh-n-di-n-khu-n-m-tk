param(
    [string]$PromptFile = "prompts/review.txt",
    [string]$OutFile = "logs/copilot.out"
)

New-Item -ItemType Directory -Force -Path logs | Out-Null

if (!(Test-Path $PromptFile)) {
    "Prompt file not found: $PromptFile" | Tee-Object $OutFile
    exit 1
}

$PromptContent = Get-Content $PromptFile -Raw

"Starting Copilot review..." | Tee-Object $OutFile
"Prompt file: $PromptFile" | Tee-Object -Append $OutFile
"--------------------------------------" | Tee-Object -Append $OutFile

try {
    $PromptContent | copilot 2>&1 | Tee-Object -Append $OutFile
    $exitCode = $LASTEXITCODE
}
catch {
    "Copilot execution failed: $_" | Tee-Object -Append $OutFile
    exit 1
}

"--------------------------------------" | Tee-Object -Append $OutFile
"Copilot finished with exit code: $exitCode" | Tee-Object -Append $OutFile

exit $exitCode