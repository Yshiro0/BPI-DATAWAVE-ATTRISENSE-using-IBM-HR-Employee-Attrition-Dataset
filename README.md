# BPI-DATAWAVE-ATTRISENSE-using-IBM-HR-Employee-Attrition-Dataset 
AttriSense is an HR analytics tool that predicts employee attrition, visualizes key insights, and provides data-driven solutions to help organizations improve retention and workforce satisfaction.

This guide walks you from a clean machine to a running dashboard with predictions and exports.

1) Prerequisites
- Windows 10/11, macOS, or Linux
- Python 3.10+ (3.11/3.12/3.13 all fine)
- Internet access for installing Python packages

Check Python:
- Windows PowerShell: python --version
- macOS/Linux: python3 --version

If Python is missing, install it from https://www.python.org/downloads/

2) Get the source code
- Place the project folder somewhere convenient (e.g., Desktop or Downloads)
- Project root should contain files like: README.md, requirements.txt, streamlit_dashboard.py, streamlit_dashboard_simple.py

3) Create and activate a virtual environment (recommended)
Windows (PowerShell):
- python -m venv .venv
- .\.venv\Scripts\Activate.ps1

macOS/Linux (bash/zsh):
- python3 -m venv .venv
- source .venv/bin/activate

4) Install dependencies
- Upgrade pip: python -m pip install --upgrade pip
- Install packages: python -m pip install -r requirements.txt

Notes:
- This installs Streamlit, scikit-learn, CatBoost, LightGBM, XGBoost, SHAP, and Excel engines (xlsxwriter/openpyxl).

5) Ensure the dataset and model exist
- Dataset file should be present at project root: WA_Fn-UseC_-HR-Employee-Attrition.csv
- A trained model is expected at submissions/hr_best_model.pkl

If you do not have hr_best_model.pkl yet:
- You can still open the dashboard and explore, but prediction features will be limited until a model exists.
- Optionally, run your training pipeline to produce a model, or copy the provided pkl into submissions/.

6) Run the dashboard
Option A (full UI):
- python streamlit_dashboard.py

Option B (simplified UI):
- python streamlit_dashboard_simple.py

Or using Streamlit directly (either file):
- streamlit run streamlit_dashboard.py
- streamlit run streamlit_dashboard_simple.py

After launch, your browser will open automatically (or visit the displayed local URL, typically http://localhost:8501).

7) Using the app
- Dashboard: Overview metrics and charts
- Predictions: Enter employee details and compute attrition risk
- Quick Actions (on Dashboard):
  - View employee records: Displays the raw CSV data in a table
  - Run attrition predictions: Runs predictions for the entire dataset and shows a preview (with probabilities and risk labels)
  - Generate Excel report: Exports an Excel workbook with three sheets:
    - RawData: the input dataset
    - Predictions: dataset + Predicted_Attrition_Proba + Predicted_Risk_Level
    - Summary: counts of risk buckets

8) Updating dependencies later
- With the venv active: python -m pip install -r requirements.txt

9) Common issues and fixes
- ModuleNotFoundError: No module named 'xlsxwriter' or 'openpyxl'
  - Run: python -m pip install xlsxwriter openpyxl
- Streamlit not found
  - Run: python -m pip install streamlit
- Wrong Python used (multiple Pythons installed)
  - On Windows, try: py -m venv .venv and py -m pip ...
  - On macOS/Linux, try: python3 instead of python
- Model not found warning
  - Copy a trained model file to submissions/hr_best_model.pkl, or run your training workflow
- Port already in use (when running Streamlit)
  - Use: streamlit run streamlit_dashboard.py --server.port 8502

10) Deactivate the virtual environment
- Windows: deactivate
- macOS/Linux: deactivate

That’s it! You should now be able to explore the dashboard, run predictions, and export Excel reports.
