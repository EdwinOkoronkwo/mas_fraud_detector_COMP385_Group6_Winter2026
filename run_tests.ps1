# ==============================================================================
# MAS Fraud Detector - Automated Test Runner (Windows PowerShell)
# This script executes the full suite of unit and integration tests.
# ==============================================================================

$ErrorActionPreference = "Stop"

Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "   🚀 Starting MAS Fraud Detector Test Suite    " -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Cyan

# 1. Environment Check
Write-Host "`n[1/3] Checking Environment..." -ForegroundColor Cyan
if (!(Get-Command pytest -ErrorAction SilentlyContinue)) {
    Write-Host "❌ pytest could not be found. Please ensure your virtual environment is active." -ForegroundColor Red
    exit 1
}
Write-Host "✅ Environment Ready." -ForegroundColor Green

# 2. Run All Tests
# This targets the 'tests' folder recursively to catch all unit and integration files
Write-Host "`n[2/3] Running Full Test Suite (Unit & Integration)..." -ForegroundColor Cyan
Write-Host "Targeting: ./tests" -ForegroundColor Gray

# Run pytest
# -v: Verbose
# -c: Use project config if available
pytest tests/ -v

$TEST_EXIT_CODE = $LastExitCode

# 3. Final Reporting
Write-Host "`n==================================================" -ForegroundColor Cyan

if ($TEST_EXIT_CODE -eq 0) {
    Write-Host "🏆 FINAL STATUS: ALL TESTS PASSED" -ForegroundColor Green
    exit 0
} else {
    Write-Host "💀 FINAL STATUS: TESTS FAILED (Exit Code: $TEST_EXIT_CODE)" -ForegroundColor Red
    exit 1
}

Write-Host "==================================================" -ForegroundColor Cyan