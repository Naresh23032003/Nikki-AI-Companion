@echo off
title Nikki RVC Worker (call voice)
cd /d "%~dp0"
echo ============================================================
echo  RVC voice worker on http://127.0.0.1:3002
echo  Keep this window open whenever call_voice = kokoro_rvc.
echo  It holds her voice model warm on the GPU for low-latency calls.
echo ============================================================
applio\env\Scripts\python.exe tools\rvc_server.py
pause
