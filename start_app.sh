#!/bin/bash

echo ""
echo "========================================"
echo "   LOAN APPROVAL PREDICTION APP"
echo "========================================"
echo ""

# Check if virtual environment exists
if [ ! -f "venv/bin/activate" ]; then
    echo "[ERROR] Virtual environment not found!"
    echo "Please first create the virtual environment:"
    echo ""
    echo "python3 -m venv venv"
    echo "source venv/bin/activate"
    echo "pip install -r requirements.txt"
    echo ""
    exit 1
fi

echo "[INFO] Activating virtual environment..."
source venv/bin/activate

echo "[INFO] Checking dependencies..."
if ! pip show streamlit &> /dev/null; then
    echo "[WARNING] Streamlit not installed. Installing..."
    pip install -r requirements.txt
fi

echo "[INFO] Launching Streamlit application..."
echo ""
echo "The application will open in your browser at:"
echo "http://localhost:8501"
echo ""
echo "To stop the application, press Ctrl+C"
echo ""

streamlit run app.py

echo ""
echo "[INFO] Application closed."
