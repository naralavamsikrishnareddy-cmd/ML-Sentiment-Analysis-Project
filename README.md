# ML-Sentiment-Analysis-Project

## AI Review Intelligence & Sentiment Analysis

A Flask-based multi-domain NLP project with a Streamlit-style dashboard recreated as a normal web application. It supports Product, Movie, and Restaurant reviews.

## Main features
- Overview dashboard with sentiment/domain distribution
- Analyze a new review
- Four-model ensemble: Naive Bayes, Logistic Regression, SVM, Random Forest-style model
- TF-IDF + text preprocessing
- Soft-voting ensemble
- Aspect Intelligence
- Batch CSV analysis and result download
- Model Evaluation
- Dataset preview and CSV upload/retraining
- Pure-Python ML engine: no NumPy, SciPy, scikit-learn, pandas, or joblib

## Why this version
The original project failed with `DLL load failed while importing sparsefuncs_fast: An Application Control policy has blocked this file.` This version avoids compiled scikit-learn dependencies, so it is suitable for the affected Windows environment.

## Run in VS Code
1. Extract the ZIP.
2. Open the `ML_Project_Clean` folder in VS Code.
3. Open Terminal.
4. Run:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python train.py
python app.py
```

5. Open **http://127.0.0.1:5000** in Chrome/Edge.

## Easier method
Double-click `run_project.bat`.

## Important
Do not copy the old `.venv` folder into this project. Create a new `.venv` using the commands above.
