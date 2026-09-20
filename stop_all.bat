@echo off
REM Stop all three services by killing whatever listens on their ports.
for %%p in (8000 4000 5173) do (
    for /f "tokens=5" %%a in ('netstat -ano ^| findstr :%%p ^| findstr LISTENING') do (
        echo Killing PID %%a on port %%p
        taskkill /PID %%a /F >nul 2>&1
    )
)
echo All services stopped.
