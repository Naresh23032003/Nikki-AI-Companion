@echo off
title RVC TensorBoard
cd /d "%~dp0"
echo Opening TensorBoard on http://localhost:6006 (loss curves live here)
start http://localhost:6006
applio\env\Scripts\python.exe -m tensorboard.main --logdir applio\logs --port 6006
