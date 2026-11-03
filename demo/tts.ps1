# One WAV per narration line, spoken by the built-in Windows voice (no service, no cost).
Add-Type -AssemblyName System.Speech
$lines = Get-Content -Raw -Encoding UTF8 "$PSScriptRoot\narration.json" | ConvertFrom-Json
$voice = New-Object System.Speech.Synthesis.SpeechSynthesizer
$voice.Rate = 0
foreach ($p in $lines.PSObject.Properties) {
    $voice.SetOutputToWaveFile("$PSScriptRoot\vo_$($p.Name).wav")
    $voice.Speak($p.Value)
}
$voice.SetOutputToNull()
$voice.Dispose()
