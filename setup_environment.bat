@echo off
setlocal
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
  echo Existing ProtocolLens environment found.
  goto install_packages
)

set "PYTHON_EXE=%LOCALAPPDATA%\Programs\Python\Python310\python.exe"
if not exist "%PYTHON_EXE%" set "PYTHON_EXE=python"

echo Creating an isolated Python environment...
"%PYTHON_EXE%" -m venv .venv
if errorlevel 1 (
  echo Unable to create the environment. Install Python 3.10-3.12 and try again.
  pause
  exit /b 1
)

:install_packages
echo Installing ProtocolLens dependencies...
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto install_failed
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto install_failed

echo.
echo Setup complete. Run run_dashboard.bat to start the demonstration.
pause
exit /b 0

:install_failed
echo.
echo Dependency installation failed. Check the internet connection and retry.
pause
exit /b 1
