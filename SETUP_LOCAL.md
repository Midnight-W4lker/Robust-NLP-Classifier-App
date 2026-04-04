---
noteId: "8df728e02ffe11f1a2cbf54eb507184d"
tags: []

---

# Local Development Setup Guide

## Prerequisites
- Python 3.10.x installed on your system
- pip package manager

## Setup Instructions

### 1. Check Python Version
```bash
python --version
# or
py -3.10 --version
```

If you don't have Python 3.10, download it from: https://www.python.org/downloads/release/python-31012/

### 2. Create Virtual Environment with Python 3.10

**Option A - If python command is 3.10:**
```bash
cd c:\Users\PC\Documents\APPS\robust_nlp_classification_app
python -m venv venv
```

**Option B - If you have multiple Python versions:**
```bash
cd c:\Users\PC\Documents\APPS\robust_nlp_classification_app
py -3.10 -m venv venv
```

**Option C - Using full path to Python 3.10:**
```bash
cd c:\Users\PC\Documents\APPS\robust_nlp_classification_app
C:\Python310\python.exe -m venv venv
```

### 3. Activate Virtual Environment

**Windows (Command Prompt):**
```cmd
venv\Scripts\activate.bat
```

**Windows (PowerShell):**
```powershell
venv\Scripts\Activate.ps1
```

**Linux/Mac:**
```bash
source venv/bin/activate
```

After activation, you should see `(venv)` in your command prompt.

### 4. Verify Python Version in Virtual Environment
```bash
python --version
# Should show: Python 3.10.x
```

### 5. Upgrade pip
```bash
python -m pip install --upgrade pip
```

### 6. Install All Requirements (Including gensim)
```bash
pip install -r requirements.txt
```

This will install all packages including:
- numpy, pandas, scikit-learn
- gensim==4.3.3 (works on Python 3.10!)
- flask, streamlit
- matplotlib, seaborn, reportlab
- joblib, requests, tqdm

### 7. Verify gensim Installation
```bash
python -c "import gensim; print(f'gensim {gensim.__version__} installed successfully!')"
```

### 8. Train Models (First Time Setup)
```bash
python train_models.py
```

This will:
- Load training data from `data/train.csv`
- Train TF-IDF, Word2Vec, and GloVe models
- Save trained models to `models/` directory
- Generate `training_results.json`

### 9. Run Streamlit App
```bash
streamlit run streamlit_app.py
```

The app will open in your browser at: http://localhost:8501

## Troubleshooting

### Issue: gensim fails to install
**Solution:** Make sure you're using Python 3.10, not 3.11+ or 3.14

### Issue: "No module named 'gensim'"
**Solution:** 
1. Activate virtual environment first
2. Run `pip install gensim==4.3.3`

### Issue: ModuleNotFoundError for other packages
**Solution:** 
```bash
pip install -r requirements.txt
```

### Issue: Models not found error
**Solution:** Train models first:
```bash
python train_models.py
```

### Issue: Virtual environment activation fails in PowerShell
**Solution:** Run PowerShell as Administrator and execute:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

## Additional Commands

### Generate Sample Datasets
```bash
python generate_datasets.py
```

### Run Flask API
```bash
python api.py
```

### Generate PDF Report
```bash
python generate_report.py
```

### Run All Scripts
```bash
python run_all.py
```

## Deactivate Virtual Environment
```bash
deactivate
```

## Notes
- The virtual environment (`venv/` folder) is gitignored
- Always activate the virtual environment before running any Python scripts
- Python 3.10 is required for gensim compatibility
- Streamlit Cloud uses Python 3.14, so gensim is optional there (only TF-IDF works)
- Locally with Python 3.10, all three models (TF-IDF, Word2Vec, GloVe) work perfectly
