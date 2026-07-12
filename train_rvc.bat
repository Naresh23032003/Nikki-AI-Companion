@echo off
title Nikki RVC Training
cd /d "%~dp0"
echo ============================================================
echo  Nikki RVC training - full chain (prerequisites, preprocess,
echo  extract, train 300 epochs, index, export)
echo  Live log also written to: rvc_training.log
echo  Loss curves: run watch_training.bat  (TensorBoard)
echo ============================================================
applio\env\Scripts\python.exe tools\train_rvc_headless.py --epochs 300 --batch 4 2>&1 | powershell -Command "$input | Tee-Object -FilePath rvc_training.log"
echo.
echo Training process ended. Check above for TRAINING COMPLETE or errors.
pause
