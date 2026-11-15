@echo off
rem Builds a single file dist\AldkLight.exe; Python is not needed on the target machine afterwards.
setlocal
cd /d "%~dp0"
if not exist build mkdir build
.venv\Scripts\python.exe -c "from aldk_light.icon import save_ico; save_ico('build/icon.ico')" || exit /b 1
.venv\Scripts\python.exe -m PyInstaller --noconfirm --clean --onefile --windowed --name AldkLight --icon "%~dp0build\icon.ico" --add-data "%~dp0aldk_light\web;aldk_light\web" --hidden-import pystray._win32 --exclude-module tkinter --specpath build --workpath build\work --distpath dist aldk_light_app.py || exit /b 1
echo.
echo Done: dist\AldkLight.exe
