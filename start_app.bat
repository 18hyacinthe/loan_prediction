@echo off
echo.
echo ========================================
echo   LOAN APPROVAL PREDICTION APP
echo ========================================
echo.

REM Check if virtual environment exists
if not exist "venv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment not found!
    echo Please first create the virtual environment:
    echo.
    echo python -m venv venv
    echo venv\Scripts\activate
    echo pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

echo [INFO] Activating virtual environment...
call venv\Scripts\activate

echo [INFO] Checking dependencies...
pip show streamlit >nul 2>&1
if errorlevel 1 (
    echo [WARNING] Streamlit not installed. Installing...
    pip install -r requirements.txt
)

echo [INFO] Launching Streamlit application...
echo.
echo The application will open in your browser at:
echo http://localhost:8501
echo.
echo To stop the application, press Ctrl+C
echo.
streamlit run app.py

echo.
echo [INFO] Application closed.
pause
