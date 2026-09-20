@echo off
REM ---------------------------------------------------------------------------
REM Parkinson Multimodal AI - start all three services in separate windows
REM   1. Python ML service  -> http://127.0.0.1:8000  (needs ml\.venv + ml\artifacts)
REM   2. Express backend    -> http://localhost:4000  (needs backend\node_modules)
REM   3. React frontend     -> http://localhost:5173  (needs frontend\node_modules)
REM ---------------------------------------------------------------------------
start "ML Service (Python :8000)" cmd /k "ml\.venv\Scripts\python.exe -m uvicorn ml.api.app:app --host 127.0.0.1 --port 8000"
timeout /t 4 /nobreak >nul
start "Backend (Express :4000)" cmd /k "cd backend && set PORT=4000&& set ML_SERVICE_URL=http://127.0.0.1:8000&& node src/server.js"
timeout /t 2 /nobreak >nul
start "Frontend (Vite :5173)" cmd /k "cd frontend && npm run dev"

echo.
echo All three services are starting in separate windows.
echo   Frontend (open this):  http://localhost:5173
echo   Backend API:           http://localhost:4000/api/health
echo   ML service:            http://127.0.0.1:8000/health
echo.
echo Close each window (or run stop_all.bat) to stop the stack.
pause
