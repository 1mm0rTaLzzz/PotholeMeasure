# End-to-end driver for the PotholeMeasure paper run on Windows PowerShell.
#
# Phases (skip with the corresponding -Skip* flag):
#   1. data       — RDD2022 VOC -> COCO + SAM2 polygon masks
#   2. download   — pretrained pothole-seg checkpoint from HuggingFace
#   3. finetune   — short fine-tune of the pretrained on RDD2022
#   4. inference  — run the full pipeline on data/processed/test
#   5. ablation   — per-recipe descriptive statistics on the same set
#   6. figures    — severity grid + hero image
#
# Example:
#   .\run_all.ps1                                     # full run
#   .\run_all.ps1 -SkipDataPrep -SkipFinetune         # quick smoke test
#   .\run_all.ps1 -HfRepo keremberke/yolov8m-pothole-segmentation -Epochs 10
#
# Output lands in $Out (default: experiments\results\paper).

param(
    [string] $RddRoot      = "data\raw\RDD2022",
    [string] $Out          = "experiments\results\paper",
    [string] $HfRepo       = "keremberke/yolov8s-pothole-segmentation",
    [int]    $Epochs       = 20,
    [double] $ConfThreshold= 0.10,
    [int]    $MaxAblation  = 200,
    [switch] $SkipDataPrep,
    [switch] $SkipMasks,
    [switch] $SkipDownload,
    [switch] $SkipFinetune,
    [switch] $SkipInference,
    [switch] $SkipAblation,
    [switch] $SkipFigures
)

$ErrorActionPreference = "Stop"

function Step($name) {
    Write-Host ""
    Write-Host "==================== $name ====================" -ForegroundColor Cyan
}

# Resolve where the downloaded weights will land.
$ckpt_dir = "experiments\checkpoints\seg"
$repo_filename = ($HfRepo -replace "/", "_") + ".pt"
$pretrained = Join-Path $ckpt_dir $repo_filename

# ----- 1. Data prep ----------------------------------------------------------
if (-not $SkipDataPrep) {
    Step "1/6 prepare_data ($RddRoot -> data\)"
    $maskFlag = @()
    if (-not $SkipMasks) { $maskFlag = @("--generate-masks") }
    python scripts/prepare_data.py --rdd-root $RddRoot --out-root data @maskFlag
    if ($LASTEXITCODE -ne 0) { throw "prepare_data failed" }
} else {
    Step "1/6 SKIPPED (data prep)"
}

# ----- 2. Download pretrained -----------------------------------------------
if (-not $SkipDownload) {
    Step "2/6 download_pretrained ($HfRepo)"
    python scripts/download_pretrained.py --hf-repo $HfRepo
    if ($LASTEXITCODE -ne 0) { throw "download_pretrained failed" }
} else {
    Step "2/6 SKIPPED (download)"
}
if (-not (Test-Path $pretrained)) {
    throw "expected weights not found: $pretrained"
}

# ----- 3. Fine-tune (recommended for paper) ---------------------------------
$weightsForInference = $pretrained
if (-not $SkipFinetune) {
    Step "3/6 fine-tune ($Epochs epochs from $pretrained)"
    # Override epochs via env (train_segmentation reads the YAML); set epochs in config.
    $cfg = Get-Content configs\default.yaml -Raw
    $cfg = [regex]::Replace($cfg, 'epochs:\s*\d+', "epochs: $Epochs")
    Set-Content -Path configs\default.yaml -Value $cfg -NoNewline
    python scripts/train_segmentation.py --init $pretrained
    if ($LASTEXITCODE -ne 0) { throw "fine-tune failed" }
    # train_segmentation copies best.pt to configs.segmentation.finetuned_weights,
    # which by default is experiments/checkpoints/seg/best.pt.
    $weightsForInference = "experiments\checkpoints\seg\best.pt"
} else {
    Step "3/6 SKIPPED (fine-tune)"
}

# ----- 4. Pipe weights + lower conf into the config -------------------------
Step "config: finetuned_weights=$weightsForInference  conf_threshold=$ConfThreshold"
$cfg = Get-Content configs\default.yaml -Raw
$weightsEsc = $weightsForInference -replace '\\', '/'
$cfg = [regex]::Replace($cfg, 'finetuned_weights:.*', "finetuned_weights: $weightsEsc")
$cfg = [regex]::Replace($cfg, 'conf_threshold:\s*[\d\.]+', "conf_threshold: $ConfThreshold")
Set-Content -Path configs\default.yaml -Value $cfg -NoNewline

# ----- 5. Inference ----------------------------------------------------------
$infOut = Join-Path $Out "inference"
if (-not $SkipInference) {
    Step "4/6 run_inference (data\processed\test -> $infOut)"
    if (-not (Test-Path "data\processed\test")) {
        throw "data\processed\test not found — re-run with data prep enabled."
    }
    python scripts/run_inference.py --input data\processed\test --output $infOut
    if ($LASTEXITCODE -ne 0) { throw "run_inference failed" }
} else {
    Step "4/6 SKIPPED (inference)"
}

# ----- 6. Relative ablation --------------------------------------------------
$ablOut = Join-Path $Out "ablation"
if (-not $SkipAblation) {
    Step "5/6 relative_ablation (-> $ablOut)"
    python scripts/relative_ablation.py --input-dir data\processed\test --output $ablOut --max-images $MaxAblation
    if ($LASTEXITCODE -ne 0) { throw "relative_ablation failed" }
} else {
    Step "5/6 SKIPPED (ablation)"
}

# ----- 7. Figures ------------------------------------------------------------
if (-not $SkipFigures) {
    Step "6/6 make_paper_figures (-> $infOut\figures)"
    python scripts/make_paper_figures.py --inference-dir $infOut
    if ($LASTEXITCODE -ne 0) { throw "make_paper_figures failed" }
} else {
    Step "6/6 SKIPPED (figures)"
}

Write-Host ""
Write-Host "==================== DONE ====================" -ForegroundColor Green
Write-Host "Inference JSON+viz : $infOut"
Write-Host "Ablation table     : $ablOut\relative_ablation.tex"
Write-Host "Severity grid      : $infOut\figures\severity_grid.jpg"
Write-Host "Hero figure        : $infOut\figures\hero.jpg"
Write-Host ""
Write-Host "For a real (GT-backed) MAE/RMSE table, hand-measure ~30 potholes," -ForegroundColor Yellow
Write-Host "save them as data\annotations\test_gt.json (schema in scripts\evaluate.py)," -ForegroundColor Yellow
Write-Host "then run: python scripts/evaluate.py --gt data\annotations\test_gt.json --output $Out\evaluate.csv --latex" -ForegroundColor Yellow
