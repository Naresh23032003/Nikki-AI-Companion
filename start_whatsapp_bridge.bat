@echo off
title Nikki WhatsApp Bridge
cd /d "%~dp0\whatsapp-bridge"
rem Your personal WhatsApp number (digits only, with country code).
rem Set it once system-wide with:  setx WA_TARGET 91XXXXXXXXXX
rem or replace the placeholder below.
if not defined WA_TARGET set WA_TARGET=91XXXXXXXXXX
rem Multiple people: comma-separated allowlist, each mapped to a persona in
rem config.yaml `profiles:`. WA_TARGET is included automatically.
rem   setx WA_TARGETS 919876543210,919999999999
if not defined WA_TARGETS set WA_TARGETS=%WA_TARGET%
echo ============================================================
echo  WhatsApp bridge starting for +%WA_TARGETS%
echo  First run: scan the QR code with the COMPANION's WhatsApp
echo  (Linked devices). After that, LocalAuth remembers the login.
echo ============================================================
node index.js
pause
