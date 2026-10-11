param(
    [switch]$Gui,
    [Parameter(Mandatory=$true)]
    [ValidatePattern('^[A-Za-z0-9_.-]+@[A-Za-z0-9_.-]+$')]
    [string]$Server,
    [ValidateRange(1, 65535)]
    [int]$Port = 22,
    [Parameter(Mandatory=$true)]
    [ValidatePattern('^/[A-Za-z0-9_./-]+$')]
    [string]$RemoteDirectory,
    [ValidatePattern('^/[A-Za-z0-9_./-]+$')]
    [string]$IsaacPython = '/isaac-sim/python.sh'
)

$ErrorActionPreference = 'Stop'
$remote = $RemoteDirectory.TrimEnd('/')
$sshOptions = @('-F', 'NUL', '-p', [string]$Port)
$scpOptions = @('-F', 'NUL', '-P', [string]$Port)

& ssh @sshOptions $server "mkdir -p $remote/scene_original"
if ($LASTEXITCODE -ne 0) { throw 'Cannot prepare remote Franka directory.' }
& scp @scpOptions (Join-Path $PSScriptRoot 'run.py') (Join-Path $PSScriptRoot 'visual.py') "${server}:${remote}/"
if ($LASTEXITCODE -ne 0) { throw 'Cannot upload Franka scripts.' }
$sceneFiles = @('scene.usda', 'layout.usda', 'bindings.json', 'scene_manifest.json') |
    ForEach-Object { Join-Path (Join-Path $PSScriptRoot 'scene_original') $_ }
& scp @scpOptions @sceneFiles "${server}:${remote}/scene_original/"
if ($LASTEXITCODE -ne 0) { throw 'Cannot upload the original tabletop scene.' }

$displayOption = if ($Gui) { '--gui' } else { '' }
& ssh @sshOptions $Server "timeout 900 $IsaacPython $remote/run.py --scene $remote/scene_original --output $remote/episode.json --capture $displayOption"
$episodeExitCode = $LASTEXITCODE
$evidence = Join-Path $PSScriptRoot 'evidence'
New-Item -ItemType Directory -Force -Path $evidence | Out-Null
& scp @scpOptions "${server}:${remote}/episode.json" $evidence
if ($LASTEXITCODE -ne 0) { throw 'Could not retrieve episode.json; inspect the server output above.' }
& scp @scpOptions -r "${server}:${remote}/episode_frames" $evidence
if ($LASTEXITCODE -ne 0) { throw 'Could not retrieve phase images; inspect the server output above.' }
if ($episodeExitCode -ne 0) { throw 'Episode failed physical acceptance; see evidence/episode.json and evidence/episode_frames/.' }
Write-Host "Success. Result: $(Join-Path $evidence 'episode.json'); images: $(Join-Path $evidence 'episode_frames')"
