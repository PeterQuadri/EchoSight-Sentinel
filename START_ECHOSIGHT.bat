@echo off
echo ==========================================
echo 🚀 STARTING ECHOSIGHT SENTINEL SYSTEM
echo ==========================================

echo [1/2] Starting AI Backend in new window...
start cmd /k "cd web_backend && .\venv\Scripts\activate && python main.py"

echo [2/2] Starting Web Dashboard in new window...
start cmd /k "cd web_dashboard && npm run dev"

echo.
echo ==========================================
echo ✅ BOTH SERVICES ARE STARTING!
echo 🌐 Open Bridge: http://localhost:3000
echo ==========================================
pause
