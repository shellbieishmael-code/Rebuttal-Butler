@echo off
setlocal
cd /d "%~dp0"
title Building Rebuttal Butler

rem ---------------------------------------------------------------------------
rem Builds a standalone Windows app: nobody who runs it needs Python, VB-Cable,
rem OBS or an internet connection.
rem
rem   build_windows.bat                 CPU build, ships small.en (recommended)
rem   build_windows.bat base.en         CPU build, smaller and faster
rem   build_windows.bat small.en gpu    adds the NVIDIA GPU libraries (about 1 GB more).
rem                                     The app still works on the CPU if the GPU is unavailable.
rem
rem Needs, on the PC doing the build only: Windows 10/11, Python 3.14 (3.12 also works) from
rem python.org, and an internet connection (to download the libraries and model).
rem Keep this file, main_standalone.py, requirements.txt and HOW_TO_USE.txt together.
rem ---------------------------------------------------------------------------

set "MODEL=small.en"
if not "%~1"=="" set "MODEL=%~1"
set "GPU=0"
if /i "%~2"=="gpu" set "GPU=1"

set "PY="
for %%V in (3.14 3.13 3.12) do (
    if not defined PY (
        py -%%V --version >nul 2>&1
        if not errorlevel 1 set "PY=py -%%V"
    )
)
if defined PY goto havepython
python --version >nul 2>&1
if not errorlevel 1 set "PY=python"
if defined PY goto havepython
echo Python was not found. Install Python 3.14 from python.org, tick "Add python.exe to PATH", then run this again.
goto fail

:havepython
echo Using: %PY%
if not exist ".buildenv\Scripts\activate.bat" %PY% -m venv .buildenv
if errorlevel 1 goto fail
call ".buildenv\Scripts\activate.bat"
python -m pip install --upgrade pip
pip install -r requirements.txt pyinstaller==6.22.3
if errorlevel 1 goto fail
if "%GPU%"=="1" (
    echo Adding the NVIDIA GPU libraries ^(large download^)...
    pip install -r requirements-gpu.txt
    if errorlevel 1 goto fail
    rem PyInstaller cannot collect NVIDIA's DLLs by package name, so the folders are named directly.
)

echo.
if not exist "assets\rebuttal_butler.ico" (
    echo Missing assets\rebuttal_butler.ico. Put the icon in the assets folder, then run this again.
    goto fail
)
if not exist "assets\rebuttal_butler_logo.png" (
    echo Missing assets\rebuttal_butler_logo.png. Put the logo in the assets folder, then run this again.
    goto fail
)
echo Building the app...
if exist "dist\RebuttalButler" rmdir /s /q "dist\RebuttalButler"
if "%GPU%"=="1" goto gpubuild
pyinstaller --noconfirm --clean --windowed --name RebuttalButler --collect-data faster_whisper --collect-binaries ctranslate2 --collect-binaries onnxruntime --hidden-import _portaudiowpatch --collect-binaries lameenc --collect-binaries miniaudio --icon "assets\rebuttal_butler.ico" --add-data "assets\rebuttal_butler_logo.png;assets" --add-data "assets\rebuttal_butler.ico;assets" main_standalone.py
if errorlevel 1 goto fail
goto built

:gpubuild
rem The NVIDIA DLLs are named directly: PyInstaller cannot collect them by package name.
pyinstaller --noconfirm --clean --windowed --name RebuttalButler --collect-data faster_whisper --collect-binaries ctranslate2 --collect-binaries onnxruntime --hidden-import _portaudiowpatch --collect-binaries lameenc --collect-binaries miniaudio --icon "assets\rebuttal_butler.ico" --add-data "assets\rebuttal_butler_logo.png;assets" --add-data "assets\rebuttal_butler.ico;assets" --add-binary ".buildenv\Lib\site-packages\nvidia\cublas\bin\*.dll;nvidia\cublas\bin" --add-binary ".buildenv\Lib\site-packages\nvidia\cudnn\bin\*.dll;nvidia\cudnn\bin" main_standalone.py
if errorlevel 1 goto fail

:built

if "%GPU%"=="1" (
    if not exist "dist\RebuttalButler\_internal\nvidia\cublas\bin\cublas64_12.dll" (
        echo.
        echo PROBLEM: the NVIDIA cuBLAS library was not put in the app. The GPU build cannot be used.
        echo Send this window to the person helping you.
        goto fail
    )
    dir /b "dist\RebuttalButler\_internal\nvidia\cudnn\bin\*.dll" >nul 2>&1
    if errorlevel 1 (
        echo.
        echo PROBLEM: the NVIDIA cuDNN library was not put in the app. The GPU build cannot be used.
        goto fail
    )
    echo GPU libraries are in the app.
)

echo.
echo Putting the %MODEL% speech model inside the app folder...
python -c "from huggingface_hub import snapshot_download as d; d('Systran/faster-whisper-%MODEL%', local_dir=r'dist\RebuttalButler\models\%MODEL%', allow_patterns=['config.json','preprocessor_config.json','model.bin','tokenizer.json','vocabulary.*'])"
if errorlevel 1 goto fail
if exist "dist\RebuttalButler\models\%MODEL%\.cache" rmdir /s /q "dist\RebuttalButler\models\%MODEL%\.cache"
if exist rebuttal_library.json copy /y rebuttal_library.json "dist\RebuttalButler\" >nul
copy /y HOW_TO_USE.txt "dist\RebuttalButler\" >nul
copy /y CheckSetup.bat "dist\RebuttalButler\" >nul

echo.
echo Checking the finished app...
if exist "dist\RebuttalButler\selftest.txt" del "dist\RebuttalButler\selftest.txt"
start /wait "" "dist\RebuttalButler\RebuttalButler.exe" --selftest
echo.
if exist "dist\RebuttalButler\selftest.txt" type "dist\RebuttalButler\selftest.txt"

echo.
set "ZIPNAME=RebuttalButler.zip"
if "%GPU%"=="1" set "ZIPNAME=RebuttalButler-GPU.zip"
echo Making dist\%ZIPNAME% to hand out...
if exist "dist\%ZIPNAME%" del "dist\%ZIPNAME%"
tar -a -c -f "dist\%ZIPNAME%" -C dist RebuttalButler

echo.
echo DONE. Give people dist\%ZIPNAME%. They unzip it anywhere and run RebuttalButler.exe.
pause
exit /b 0

:fail
echo.
echo The build stopped because of the problem shown above.
pause
exit /b 1
