param(
    [string]$PromptFile = "prompts/implement.txt",
    [string]$OutFile = "logs/codex.out"
)

New-Item -ItemType Directory -Force -Path logs | Out-Null

if (!(Test-Path $PromptFile)) {
    "Prompt file not found: $PromptFile" | Tee-Object $OutFile
    exit 1
}

$PromptContent = Get-Content $PromptFile -Raw

"Starting Codex execution..." | Tee-Object $OutFile
"Prompt file: $PromptFile" | Tee-Object -Append $OutFile
"--------------------------------------" | Tee-Object -Append $OutFile

try {
    codex exec "$PromptContent" 2>&1 | Tee-Object -Append $OutFile
    $exitCode = $LASTEXITCODE
}
catch {
    "Codex execution failed: $_" | Tee-Object -Append $OutFile
    exit 1
}

"--------------------------------------" | Tee-Object -Append $OutFile
"Codex finished with exit code: $exitCode" | Tee-Object -Append $OutFile

exit $exitCode