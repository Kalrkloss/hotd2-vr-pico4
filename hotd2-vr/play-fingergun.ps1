<#
  HOTD2 with webcam finger guns.

  Starts Flycast (this fork) with a light gun on port A, a red crosshair and the UDP
  light gun input on 127.0.0.1:27015, then the webcam tracker. Close the tracker
  window (q) when done; Flycast stays open until you close it. Running it again while
  either is still open reuses what is running instead of starting a second copy:
  two trackers would both steer the crosshair.

  .\play-fingergun.ps1              one player
  .\play-fingergun.ps1 -Players 2   two players (left hand on camera = P1)
  .\play-fingergun.ps1 -Camera 1    another webcam

  Needs the Windows build (build-win.cmd) and your own dump of the game (.cue/.gdi/.chd)
  in a "game" folder next to this repository, or in $env:HOTD2VR_DATA\game. The tracker
  runs from fingergun\.venv (python -m venv fingergun\.venv, then
  fingergun\.venv\Scripts\pip install -r fingergun\requirements.txt) and needs MediaPipe's
  hand_landmarker.task in fingergun\.
#>
param(
	[int]$Players = 1,
	[int]$Camera = 0,
	[int]$Port = 27015
)
$repo = Split-Path -Parent $PSScriptRoot
$data = if ($env:HOTD2VR_DATA) { $env:HOTD2VR_DATA } else { Split-Path -Parent $repo }
$exe = Join-Path $repo 'build-win\flycast.exe'
$cue = Get-ChildItem (Join-Path $data 'game') -Include *.cue, *.gdi, *.chd -Recurse | Select-Object -First 1
if (-not $cue) {
	Write-Host "`nGeen game gevonden in $(Join-Path $data 'game') / no game found there.`n"
	exit 1
}
$runDir = Join-Path $data 'run'
New-Item -ItemType Directory -Force $runDir | Out-Null
$py = Join-Path $PSScriptRoot 'fingergun\.venv\Scripts\python.exe'
if (-not (Test-Path $py)) { $py = 'python' }
$tracker = Join-Path $PSScriptRoot 'fingergun\fingergun.py'

$running = Get-CimInstance Win32_Process -Filter "Name='python.exe' OR Name='flycast.exe'" | Select-Object Name, CommandLine
if ($running | Where-Object { $_.CommandLine -like "*$tracker*" }) {
	Write-Host "`nDe vingerpistool-tracker draait al - gebruik dat venster.`n"
	exit 1
}

if ($running | Where-Object { $_.CommandLine -like "*$exe*" }) {
	Write-Host 'Flycast draait al; alleen de tracker wordt gestart.'
} else {
	# Crosshair colours are ABGR: opaque red, opaque cyan
	$cfg = "input:device1=7,config:rend.CrossHairColor1=-16776961,config:vr.UdpPort=$Port,log:LogToFile=yes"
	if ($Players -ge 2) { $cfg += ',input:device2=7,config:rend.CrossHairColor2=-256' }
	Start-Process -FilePath $exe -ArgumentList @('-config', $cfg, "`"$($cue.FullName)`"") -WorkingDirectory $runDir
}
& $py $tracker --players $Players --camera $Camera --port $Port
