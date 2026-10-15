@echo off
rem Double-click to test that the app works on this PC. It checks the speech engine,
rem the bundled model and the desktop-audio capture, and shows what it found.
rem Play some sound (or the call) while it runs so the audio check can hear something.
cd /d "%~dp0"
if exist selftest.txt del selftest.txt
start /wait "" "RebuttalButler.exe" --selftest
echo.
if exist selftest.txt type selftest.txt
echo.
pause
