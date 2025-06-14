# run_project.ps1
Param (
    [string]$RawDataDir = "data\raw",
    [string]$AllCsv = "data\raw\all_movies.csv",
    [string]$CleanCsv = "data\processed\clean.csv",
    [string]$TrainCsv = "data\processed\train.csv",
    [string]$TestCsv = "data\processed\test.csv",
    [int]$TestSize = 0.25,
    [string]$FeatCsv = "data\processed\features.csv",
    [string]$ModelOut = "models\best_model.pkl",
    [string]$Launcher = "uv"
)

function Run-Step {
    param (
        [string]$Message,
        [string]$Command
    )

    Write-Host "`n$Message" -ForegroundColor Cyan
    Invoke-Expression $Command

    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ Step failed. Exiting script." -ForegroundColor Red
        exit 1
    }
}

Run-Step "Installing Python dependencies with $Launcher sync…" "$Launcher sync"

Run-Step "Combining raw CSV files → $AllCsv" "$Launcher run .\src\data_loader.py --input `"$RawDataDir`" --output `"$AllCsv`""

Run-Step "Preprocessing raw data → $CleanCsv" "$Launcher run .\src\preprocess.py --input `"$AllCsv`" --output `"$CleanCsv`""

Run-Step "Building features → $FeatCsv" "$Launcher run .\src\features.py --input `"$CleanCsv`" --output `"$FeatCsv`""

Run-Step "Training model → $ModelOut" "$Launcher run .\src\train.py --data `"$FeatCsv`" --model-out `"$ModelOut`""

Run-Step "Evaluating model performance on features" "$Launcher run .\src\evaluate.py --model-path `"$ModelOut`" --test-data `"$FeatCsv`""

Write-Host "`n✅ All steps completed successfully!" -ForegroundColor Green

