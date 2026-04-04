@echo off
REM Setup script for Windows - Creates Python 3.10 virtual environment and installs dependencies

echo ========================================
echo NLP Classifier Local Setup
echo ========================================
echo.

REM Check if Python 3.10 is available
echo Checking for Python 3.10...
py -3.10 --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python 3.10 not found!
    echo Please install Python 3.10 from: https://www.python.org/downloads/release/python-31012/
    pause
    exit /b 1
)

echo Found: 
py -3.10 --version
echo.

REM Create virtual environment
echo Creating virtual environment with Python 3.10...
if exist venv (
    echo Virtual environment already exists. Skipping creation.
) else (
    py -3.10 -m venv venv
    if %errorlevel% neq 0 (
        echo ERROR: Failed to create virtual environment!
        pause
        exit /b 1
    )
    echo Virtual environment created successfully!
)
echo.

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat
if %errorlevel% neq 0 (
    echo ERROR: Failed to activate virtual environment!
    pause
    exit /b 1
)
echo.

REM Verify Python version
echo Verifying Python version in virtual environment...
python --version
echo.

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip
echo.

REM Install requirements
echo Installing requirements (this may take a few minutes)...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo ERROR: Failed to install requirements!
    pause
    exit /b 1
)
echo.

REM Verify gensim installation
echo Verifying gensim installation...
python -c "import gensim; print(f'✓ gensim {gensim.__version__} installed successfully!')"
if %errorlevel% neq 0 (
    echo WARNING: gensim verification failed!
) else (
    echo.
)

echo ========================================
echo Setup Complete!
echo ========================================
echo.
echo Next steps:
echo   1. Keep virtual environment activated
echo   2. Train models: python train_models.py
echo   3. Run app: streamlit run streamlit_app.py
echo.
echo To activate virtual environment later, run:
echo   venv\Scripts\activate.bat
echo.
pause
