# End-to-end PothRGBD evaluation for the paper (Windows PowerShell).
#
# Pipeline:
#   1. import_pothrgbd: convert Kaggle download into our test layout +
#      emit data/pothrgbd/test_gt.json with REAL depth/area ground truth.
#   2. point configs/default.yaml at the D415 calibration we just wrote.
#   3. run_inference on data/pothrgbd/processed/<split> to produce
#      JSON+viz with model predictions.
#   4. evaluate.py: compare every recipe against test_gt.json. This is
#      the headline Table 1 (MAE / RMSE / severity F1) for the paper.
#   5. relative_ablation + figures + benchmark on the same split.
#
# Example:
#   .\run_pothrgbd.ps1 -Src "C:\datasets\pothrgbd"
#   .\run_pothrgbd.ps1 -Src "C:\datasets\pothrgbd" -SkipImport -SkipBenchmark

param(
    [string] $Src        = "data\raw\pothrgbd",
    [string] $Out        = "experiments\results\pothrgbd",
    [string] $Calib      = "data\pothrgbd\calibration.yaml",
    [int[]]  $TargetSize = @(),
    [switch] $SkipImport,
    [switch] $SkipInference,
    [switch] $SkipEvaluate,
    [switch] $SkipAblation,
    [switch] $SkipFigures,
    [switch] $SkipBenchmark,
    [int]    $BenchmarkN = 100
)

$ErrorActionPreference = "Stop"

function Step($name) {
    Write-Host ""
    Write-Host ("==================== " + $name + " ====================") -ForegroundColor Cyan
}

# ----- 1. import + GT JSON ---------------------------------------------------
if (-not $SkipImport) {
    Step ("1/6 import_pothrgbd from " + $Src)
    $tsArgs = @()
    if ($TargetSize.Count -eq 2) {
        $tsArgs = @("--target-size", $TargetSize[0], $TargetSize[1])
    }
    python scripts/import_pothrgbd.py --src $Src --out-root data\pothrgbd @tsArgs
    if ($LASTEXITCODE -ne 0) { throw "import_pothrgbd failed" }
} else {
    Step "1/6 SKIPPED (import)"
}

# ----- 2. point config at the D415 calibration -------------------------------
Step "config: paths.calibration => $Calib"
$cfg = Get-Content configs\default.yaml -Raw
$calibEsc = $Calib -replace '\\', '/'
$cfg = [regex]::Replace($cfg, 'calibration:.*', ("calibration: " + $calibEsc))
Set-Content -Path configs\default.yaml -Value $cfg -NoNewline

# ----- 3. inference ----------------------------------------------------------
$infOut = Join-Path $Out "inference"
if (-not $SkipInference) {
    $testDir = "data\pothrgbd\processed\test"
    if (-not (Test-Path $testDir)) { $testDir = "data\pothrgbd\processed\all" }
    if (-not (Test-Path $testDir)) { throw "no processed/test or processed/all under data\pothrgbd" }
    Step ("2/6 run_inference (" + $testDir + " => " + $infOut + ")")
    python scripts/run_inference.py --input $testDir --output $infOut
    if ($LASTEXITCODE -ne 0) { throw "run_inference failed" }
} else {
    Step "2/6 SKIPPED (inference)"
}

# ----- 4. evaluate against GT ------------------------------------------------
$evalOut = Join-Path $Out "evaluate.csv"
if (-not $SkipEvaluate) {
    Step ("3/6 evaluate (vs data\pothrgbd\test_gt.json => " + $evalOut + ")")
    python scripts/evaluate.py --gt data\pothrgbd\test_gt.json --output $evalOut --latex
    if ($LASTEXITCODE -ne 0) { throw "evaluate failed" }
} else {
    Step "3/6 SKIPPED (evaluate)"
}

# ----- 5. relative ablation --------------------------------------------------
$ablOut = Join-Path $Out "ablation"
if (-not $SkipAblation) {
    $testDir = "data\pothrgbd\processed\test"
    if (-not (Test-Path $testDir)) { $testDir = "data\pothrgbd\processed\all" }
    Step ("4/6 relative_ablation (out: " + $ablOut + ")")
    python scripts/relative_ablation.py --input-dir $testDir --output $ablOut
    if ($LASTEXITCODE -ne 0) { throw "relative_ablation failed" }
} else {
    Step "4/6 SKIPPED (ablation)"
}

# ----- 6. figures ------------------------------------------------------------
if (-not $SkipFigures) {
    Step ("5/6 make_paper_figures (out: " + $infOut + "\figures)")
    python scripts/make_paper_figures.py --inference-dir $infOut
    if ($LASTEXITCODE -ne 0) { throw "make_paper_figures failed" }
} else {
    Step "5/6 SKIPPED (figures)"
}

# ----- 7. benchmark ----------------------------------------------------------
$benchOut = Join-Path $Out "benchmark.json"
if (-not $SkipBenchmark) {
    $testDir = "data\pothrgbd\processed\test"
    if (-not (Test-Path $testDir)) { $testDir = "data\pothrgbd\processed\all" }
    Step ("6/6 benchmark (" + $BenchmarkN + " images)")
    python scripts/benchmark.py --input-dir $testDir --num $BenchmarkN --output $benchOut
    if ($LASTEXITCODE -ne 0) { throw "benchmark failed" }
} else {
    Step "6/6 SKIPPED (benchmark)"
}

Write-Host ""
Write-Host "==================== DONE ====================" -ForegroundColor Green
Write-Host ("Inference JSON+viz : " + $infOut)
Write-Host ("Evaluate CSV/LaTeX : " + $evalOut + "  (and .tex)")
Write-Host ("Ablation table     : " + $ablOut + "\relative_ablation.tex")
Write-Host ("Severity grid      : " + $infOut + "\figures\severity_grid.jpg")
Write-Host ("Area distribution  : " + $infOut + "\figures\area_distribution.png")
Write-Host ("Benchmark report   : " + $benchOut)
