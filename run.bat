@echo off
rem Runs Aldk Light from sources without a console window. Add --tray to start minimized to the tray.
start "" "%~dp0.venv\Scripts\pythonw.exe" "%~dp0aldk_light_app.py" %*
