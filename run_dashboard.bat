@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo ProtocolLens virtual environment was not found.
  echo Run setup_environment.bat once, then start this file again.
  pause
  exit /b 1
)

if not exist "data\processed\trials_with_features.csv" (
  echo Generated ProtocolLens data was not found.
  echo Follow the reproduction steps in README.md first.
  pause
  exit /b 1
)

echo Starting ProtocolLens at http://localhost:8501
".venv\Scripts\python.exe" -m streamlit run app.py --browser.gatherUsageStats false --server.fileWatcherType none
