<#
.SYNOPSIS
    Builds the TensorRT engines and prompt cache for the accelerated SAM 3 provider.

.DESCRIPTION
        .\scripts\build_dart_engines.ps1                 # build whatever is missing
        .\scripts\build_dart_engines.ps1 -Status         # what exists, and does it load
        .\scripts\build_dart_engines.ps1 -Force          # rebuild everything
        .\scripts\build_dart_engines.ps1 -Phrases "metal duct","ceiling fan"
                                                         # extend the text cache only

    DART (arXiv 2603.11441, cloned into third_party/DART) turns the SAM 3 weights the
    `sam3` provider already uses into two TensorRT FP16 engines plus a cached
    text encoder. This script drives DART's own export tools and puts the
    products where src/vision/sam3_trt.py looks for them:

        data/dart/hf_backbone_1008_fp16.engine     ViT-H backbone + FPN neck
        data/dart/enc_dec_1008_c4_fp16.engine      6+6 encoder-decoder, 4 classes a pass
        data/dart/text_cache.pt                    phrase -> text features + mask
        data/dart/pos_enc_1008.pt                  the encoder's position encoding
        data/dart/engines.json                     what was built, with what, on which GPU

    Everything is idempotent: a product that exists is kept unless -Force.
    The ONNX intermediates stay in data/dart/onnx because they are portable
    across TensorRT versions where the engines are not -- a TensorRT upgrade
    costs an engine rebuild (minutes), not a re-export (longer, and a 3 GB
    checkpoint load).

    DART runs in its own Python 3.12 environment at .tools/dart-venv (created
    here), never in the main venv. The main venv receives exactly one
    package from this: the tensorrt runtime, at the same pin. See
    the README for why.

    Engines are specific to the GPU model and the TensorRT version. They are
    never committed; a fresh clone runs this script once.

.PARAMETER Status
    Report what exists, its size, the manifest, and whether the main venv
    can load the engines. Exits non-zero if the provider could not run.

.PARAMETER Force
    Rebuild every product, including the ONNX exports.

.PARAMETER ImageSize
    Working resolution, divisible by 14. 1008 is what SAM 3 was trained at and
    what DART's accuracy figures are for. Must match between engines.

.PARAMETER MaxClasses
    Phrases per encoder-decoder pass. Four fits the 8 GB laptop GPU at build
    time (DART: roughly 4 GB of VRAM per class during the build); longer prompt
    lists are chunked at run time.

.PARAMETER Image
    A photograph for the backbone export's own validation pass (it compares
    the engine's features with PyTorch's). One of the bench fixtures, so the
    cosine similarity recorded in the manifest is for the kind of image we
    actually run.

.PARAMETER Phrases
    Extra phrases to encode into the text cache, beyond the concept vocabulary
    in app/vision/concepts.py and the open-vocabulary defaults.
#>
[CmdletBinding()]
param(
    [switch]$Status,
    [switch]$Force,
    [int]$ImageSize = 1008,
    [int]$MaxClasses = 4,
    [string]$Image = 'data\fixtures\photos\mep\09_2012-10-15_15-39-47_Portugal_Azores_Ribeira_Grande_JPG.jpg',
    [string[]]$Phrases = @()
)

$ErrorActionPreference = 'Stop'
$ProgressPreference    = 'SilentlyContinue'

$DoStatus = [bool]$Status
$DoForce  = [bool]$Force

# Everything stays inside the repository folder: engines, ONNX files, venvs.
$Root = Split-Path $PSScriptRoot -Parent
if (-not $env:DART_DIR) { $env:DART_DIR = Join-Path $Root 'data\dart' }
$env:PYTHONUTF8 = '1'   # TensorRT and DART print characters cp1252 cannot encode

if ($ImageSize % 14 -ne 0) { throw "-ImageSize must be divisible by 14 (ViT patch size), got $ImageSize" }

$Dart         = Join-Path $Root 'third_party\DART'   # git clone of the DART repository
$VenvDir      = Join-Path $Root '.tools\dart-venv'
$DartPython   = Join-Path $VenvDir 'Scripts\python.exe'
$ApiPython    = Join-Path $Root '.venv\Scripts\python.exe'
$ApiDir       = Join-Path $Root 'src'
$Support      = Join-Path $PSScriptRoot 'dart\build_support.py'
$Requirements = Join-Path $PSScriptRoot 'dart\requirements.txt'

$Out   = $env:DART_DIR
$Onnx  = Join-Path $Out 'onnx'
$Logs  = Join-Path $Out 'logs'
foreach ($dir in @($Out, $Onnx, $Logs)) {
    if (-not (Test-Path $dir)) { New-Item -ItemType Directory -Force -Path $dir | Out-Null }
}

# The one pin that must agree between the two environments.
$TrtPin = @(Get-Content $Requirements | Where-Object { $_ -match '^tensorrt-cu12==' })[0]
if (-not $TrtPin) { throw "scripts\dart\requirements.txt no longer pins tensorrt-cu12" }
$NvidiaIndex = 'https://pypi.nvidia.com'

$BackboneOnnxDir = Join-Path $Onnx "backbone_$ImageSize"
$BackboneOnnx    = Join-Path $BackboneOnnxDir 'hf_backbone.onnx'
$BackboneEngine  = Join-Path $Out  "hf_backbone_${ImageSize}_fp16.engine"
$EncDecOnnx      = Join-Path $Onnx "enc_dec_${ImageSize}_c${MaxClasses}.onnx"
$EncDecEngine    = Join-Path $Out  "enc_dec_${ImageSize}_c${MaxClasses}_fp16.engine"
$TextCache       = Join-Path $Out  'text_cache.pt'
$PosEnc          = Join-Path $Out  "pos_enc_$ImageSize.pt"
$Manifest        = Join-Path $Out  'engines.json'

$Products = [ordered]@{
    'backbone engine' = $BackboneEngine
    'enc-dec engine'  = $EncDecEngine
    'text cache'      = $TextCache
    'position enc'    = $PosEnc
    'manifest'        = $Manifest
}

function Write-Step($text) { Write-Host "`n==> $text" -ForegroundColor Cyan }
function Write-Ok  ($text) { Write-Host "    $text" -ForegroundColor Green }
function Write-Info($text) { Write-Host "    $text" -ForegroundColor DarkGray }
function Write-Warn($text) { Write-Host "    $text" -ForegroundColor Yellow }

function Format-Size([long]$bytes) {
    if ($bytes -ge 1GB) { return ('{0:N2} GB' -f ($bytes / 1GB)) }
    if ($bytes -ge 1MB) { return ('{0:N1} MB' -f ($bytes / 1MB)) }
    return ('{0:N0} KB' -f ($bytes / 1KB))
}

function Invoke-Dart {
    <#  Run something under the DART venv, with DART importable and its
        scripts' relative paths resolving inside its checkout. Output is
        both shown and kept: a twenty-minute engine build that fails wants
        its log read, not re-run. #>
    param([string]$Log, [string[]]$Arguments)

    $env:PYTHONPATH = $Dart
    Push-Location $Dart
    # Merged stderr arrives as ErrorRecords, and under 'Stop' the first
    # TensorRT warning would abort a build that was going fine. The exit
    # code is the verdict here, not the stream a line came from.
    $previous = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        & $DartPython @Arguments 2>&1 | Tee-Object -FilePath $Log
        $code = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $previous
        Pop-Location
        Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
    }
    if ($code -ne 0) { throw "failed (exit $code) -- see $Log" }
}

# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------
function Show-Status {
    Write-Step "DART products in $Out"
    $missing = 0
    foreach ($name in $Products.Keys) {
        $path = $Products[$name]
        if (Test-Path $path) {
            Write-Host ('    {0,-16} {1,10}  {2}' -f $name, (Format-Size (Get-Item $path).Length), (Split-Path -Leaf $path)) -ForegroundColor Green
        } else {
            Write-Host ('    {0,-16} {1,10}  {2}' -f $name, 'missing', (Split-Path -Leaf $path)) -ForegroundColor Red
            $missing++
        }
    }

    if (Test-Path $Manifest) {
        $m = Get-Content $Manifest -Raw | ConvertFrom-Json
        Write-Host ''
        Write-Info ("built {0}  tensorrt {1}  torch {2}  gpu {3}" -f $m.built_at, $m.tensorrt, $m.torch, $m.gpu)
        Write-Info ("imgsz {0}  max_classes {1}  dart {2}" -f $m.imgsz, $m.max_classes, $m.dart_commit)
        if ($m.backbone_cosine) {
            $cos = ($m.backbone_cosine.PSObject.Properties | ForEach-Object { '{0}={1}' -f $_.Name, $_.Value }) -join '  '
            Write-Info "backbone vs PyTorch cosine: $cos"
        }
    }

    Write-Host ''
    Write-Step 'TensorRT runtimes'
    foreach ($pair in @(@('dart-venv', $DartPython), @('api venv', $ApiPython))) {
        $label, $py = $pair
        if (-not (Test-Path $py)) { Write-Warn ('{0,-10} no interpreter' -f $label); continue }
        $ver = & $py -c "import tensorrt; print(tensorrt.__version__)" 2>$null
        if ($LASTEXITCODE -eq 0) { Write-Ok ('{0,-10} tensorrt {1}' -f $label, $ver) }
        else { Write-Warn ('{0,-10} tensorrt not installed' -f $label) }
    }

    # The verdict that matters: can the main venv's provider stand up on this?
    Write-Host ''
    Write-Step 'Provider check (API venv)'
    $probe = Join-Path $ApiDir 'vision\sam3_trt.py'
    if (-not (Test-Path $probe)) {
        Write-Warn 'src/vision/sam3_trt.py not present -- nothing to check yet'
    } else {
        Push-Location $ApiDir
        try {
            & $ApiPython -c "from vision.sam3_trt import engine_status; import sys; s = engine_status(); print(s.message); sys.exit(0 if s.ok else 1)"
            $code = $LASTEXITCODE
        }
        finally { Pop-Location }
        if ($code -ne 0) { $missing++ }
    }

    Write-Host ''
    if ($missing -gt 0) {
        Write-Host "  incomplete -- run .\scripts\build_dart_engines.ps1" -ForegroundColor Red
        exit 1
    }
    Write-Host '  OK: the accelerated SAM 3 provider can run.' -ForegroundColor Green
}

if ($DoStatus) { Show-Status; return }

# ---------------------------------------------------------------------------
# 1. Environments
# ---------------------------------------------------------------------------
Write-Step "DART environment ($VenvDir)"
if (-not (Test-Path $DartPython)) {
    & uv venv $VenvDir --python 3.12 --seed
    if ($LASTEXITCODE -ne 0) { throw 'uv venv failed' }
}
Write-Info 'torch cu126 (pinned; see scripts\dart\requirements.txt for why)'
& uv pip install --python $DartPython --index-url https://download.pytorch.org/whl/cu126 'torch==2.7.1' 'torchvision==0.22.1'
if ($LASTEXITCODE -ne 0) { throw 'torch install failed' }
Write-Info 'DART dependencies'
& uv pip install --python $DartPython -r $Requirements --extra-index-url $NvidiaIndex
if ($LASTEXITCODE -ne 0) { throw 'DART dependency install failed' }

Write-Step "TensorRT runtime for the API ($TrtPin)"
& uv pip install --python $ApiPython --extra-index-url $NvidiaIndex $TrtPin
if ($LASTEXITCODE -ne 0) { throw 'tensorrt install into .venv failed' }

# ---------------------------------------------------------------------------
# 2. Meta's checkpoint
# ---------------------------------------------------------------------------
Write-Step "Meta checkpoint (sam3.pt)"
$env:PYTHONPATH = $Dart
$Checkpoint = (& $DartPython $Support checkpoint | Select-Object -Last 1)
$code = $LASTEXITCODE
Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
if ($code -ne 0 -or -not (Test-Path $Checkpoint)) { throw 'could not obtain sam3.pt' }
Write-Ok $Checkpoint

# ---------------------------------------------------------------------------
# 3. Backbone engine (HF export path -- the one with correct FP16)
# ---------------------------------------------------------------------------
$ImageAbs = if ([System.IO.Path]::IsPathRooted($Image)) { $Image } else { Join-Path $Root $Image }
if (-not (Test-Path $ImageAbs)) { throw "validation image not found: $ImageAbs" }

Write-Step "Backbone engine ($ImageSize px)"
if ((Test-Path $BackboneEngine) -and -not $DoForce) {
    Write-Ok "present  $(Split-Path -Leaf $BackboneEngine)"
} else {
    New-Item -ItemType Directory -Force -Path $BackboneOnnxDir | Out-Null
    $arguments = @(
        (Join-Path $Dart 'scripts\export_hf_backbone.py'),
        '--image', $ImageAbs,
        '--imgsz', $ImageSize,
        '--output-onnx', $BackboneOnnx,
        '--output-engine', $BackboneEngine
    )
    if ((Test-Path $BackboneOnnx) -and -not $DoForce) {
        Write-Info 'ONNX present, skipping export'
        $arguments += '--skip-export'
    }
    Invoke-Dart -Log (Join-Path $Logs 'backbone.log') -Arguments $arguments
    Write-Ok "built    $(Split-Path -Leaf $BackboneEngine)"
}

# ---------------------------------------------------------------------------
# 4. Encoder-decoder engine
# ---------------------------------------------------------------------------
Write-Step "Encoder-decoder engine ($ImageSize px, $MaxClasses classes a pass)"
if ((Test-Path $EncDecEngine) -and -not $DoForce) {
    Write-Ok "present  $(Split-Path -Leaf $EncDecEngine)"
} else {
    if ((Test-Path $EncDecOnnx) -and -not $DoForce) {
        Write-Info 'ONNX present, skipping export'
    } else {
        # --presence: the eager provider's scores are sigmoid(logit) x
        # sigmoid(presence), so the engine must produce the presence logit
        # too or the two paths score on different scales.
        Invoke-Dart -Log (Join-Path $Logs 'enc_dec_export.log') -Arguments @(
            '-m', 'sam3.trt.export_enc_dec',
            '--checkpoint', $Checkpoint,
            '--output', $EncDecOnnx,
            '--max-classes', $MaxClasses,
            '--imgsz', $ImageSize,
            '--presence'
        )
    }
    # --mixed-precision none: DART's auto rule is for the backbone and is
    # wrong for this graph (their README, "Encoder-Decoder Export").
    Invoke-Dart -Log (Join-Path $Logs 'enc_dec_build.log') -Arguments @(
        '-m', 'sam3.trt.build_engine',
        '--onnx', $EncDecOnnx,
        '--output', $EncDecEngine,
        '--type', 'enc-dec',
        '--fp16',
        '--mixed-precision', 'none',
        '--max-classes', $MaxClasses
    )
    Write-Ok "built    $(Split-Path -Leaf $EncDecEngine)"
}

# ---------------------------------------------------------------------------
# 5. Text cache + position encoding
# ---------------------------------------------------------------------------
Write-Step 'Text cache (concept vocabulary + defaults) and position encoding'
$PhraseFile = Join-Path $Logs 'phrases.json'
Push-Location $ApiDir
try {
    # The vocabulary lives in the API; asking it is the only way the cache
    # cannot drift from concepts.py.
    # An empty array piped into ConvertTo-Json produces nothing at all, not
    # "[]"; the python side needs a JSON list either way.
    $extra = if ($Phrases.Count -gt 0) {
        ConvertTo-Json -InputObject @($Phrases) -Compress -AsArray
    } else { '[]' }
    & $ApiPython -c @"
import json, sys
from vision.concepts import CONCEPTS
from vision.openvocab import default_prompts
phrases = []
for concept in CONCEPTS.values():
    phrases.append(concept.name)
    phrases.extend(concept.phrasings)
phrases.extend(default_prompts())
phrases.extend(json.loads(sys.argv[1]))
seen = []
for p in phrases:
    p = p.strip().lower()
    if p and p not in seen:
        seen.append(p)
json.dump(seen, open(sys.argv[2], 'w', encoding='utf-8'), indent=1)
print(f'{len(seen)} phrases')
"@ $extra $PhraseFile
    if ($LASTEXITCODE -ne 0) { throw 'could not collect phrases from the API venv' }
}
finally { Pop-Location }

$textArgs = @(
    $Support, 'text-cache',
    '--checkpoint', $Checkpoint,
    '--phrases', $PhraseFile,
    '--out', $TextCache,
    '--pos-enc-out', $PosEnc,
    '--imgsz', $ImageSize
)
if ($DoForce) { $textArgs += '--force' }
Invoke-Dart -Log (Join-Path $Logs 'text_cache.log') -Arguments $textArgs

# ---------------------------------------------------------------------------
# 6. Manifest
# ---------------------------------------------------------------------------
Write-Step 'Manifest'
Invoke-Dart -Log (Join-Path $Logs 'manifest.log') -Arguments @(
    $Support, 'manifest',
    '--dir', $Out,
    '--dart', $Dart,
    '--imgsz', $ImageSize,
    '--max-classes', $MaxClasses,
    '--backbone-log', (Join-Path $Logs 'backbone.log'),
    '--files', (Split-Path -Leaf $BackboneEngine), (Split-Path -Leaf $EncDecEngine),
               (Split-Path -Leaf $TextCache), (Split-Path -Leaf $PosEnc)
)

Show-Status
