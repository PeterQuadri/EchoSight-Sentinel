# 🚀 How to Start EchoSight Sentinel

Since I've stopped all background processes to free up your camera, here is how you can start the system yourself whenever you need it.

### Option 1: Using the Automated Script (Recommended)
I have created a file called `START_ECHOSIGHT.bat` in your project folder. 
1.  **Double-click** `START_ECHOSIGHT.bat`.
2.  It will open two terminal windows: one for the **AI Backend** and one for the **Web Dashboard**.
3.  Once both are ready, go to [http://localhost:3000](http://localhost:3000).

---

### Option 2: Manual Start (Step-by-Step)

#### 1. Start the AI Backend
Open a terminal in `d:\DOCUMENTS\RAIN\AIML\Second_semester\Project\web_backend` and run:
```powershell
.\venv\Scripts\activate; python main.py
```
*(Wait until you see "Uvicorn running on http://0.0.0.0:8000")*

#### 2. Start the Web Dashboard
Open a **new** terminal in `d:\DOCUMENTS\RAIN\AIML\Second_semester\Project\web_dashboard` and run:
```powershell
npm run dev
```
*(Wait until you see "Ready in ...")*

#### 3. Access the System
Open your browser to: **[http://localhost:3000](http://localhost:3000)**

---

### 💡 Troubleshooting
*   **"Camera in use"**: Make sure you haven't accidentally left `realtime_detection_system.py` or a Zoom/Teams call running in the background.
*   **Port 8000/3000 busy**: This happens if a previous session didn't close properly. Restarting your computer will always fix this.
