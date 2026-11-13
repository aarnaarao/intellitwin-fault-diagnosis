# IntelliTwin – Intelligent Machine Fault Diagnosis and Failure Prediction

An AI-powered machine fault diagnosis system that analyses industrial pump sensor data and classifies operating conditions using machine learning.

## Overview

IntelliTwin is a machine learning-based predictive maintenance application designed to identify abnormal operating conditions in industrial pumps. It analyses historical sensor readings, extracts relevant features and uses a trained classification model to identify potential machine faults.

The system integrates an XGBoost classification model with a FastAPI backend and a React-based dashboard to provide machine-level diagnosis and fleet monitoring.

## Key Features

- **Machine Fleet Monitoring:** Monitor the operating conditions of 50 industrial pumps.
- **AI-Based Fault Diagnosis:** Classify pump conditions into Normal, Overload, Pressure_drop and Shutdown.
- **Priority Fault Alerts:** Highlight machines requiring attention.
- **Model Intelligence:** View model performance and compare different classification algorithms.
- **Individual Machine Analysis:** Select a pump and view its predicted fault category and class probabilities.
- **Interactive Dashboard:** Access machine information and diagnosis results through a web interface.

## Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn, XGBoost |
| Imbalance Handling | SMOTE |
| Backend | FastAPI |
| Frontend | React, Vite |
| Model Storage | Joblib |
| Development Environment | Visual Studio Code |

## Dataset

The project uses an industrial pump predictive maintenance dataset containing:

- 36,500 daily observations
- 50 industrial pumps
- 35 original dataset columns
- Sensor and operational parameters such as pressure, flow, temperature, vibration, rotation speed, power and operating hours.
- 39 engineered features used by the final model.

The target classes are Normal, Overload, Pressure_drop and Shutdown.

## Machine Learning Workflow

1. Dataset loading and exploration.
2. Data cleaning and preprocessing.
3. Feature engineering and extraction.
4. Pump-wise data splitting to reduce data leakage.
5. Handling class imbalance using different strategies, including SMOTE.
6. Training and evaluation of Random Forest, XGBoost and SVM.
7. Selection of the best validation configuration.
8. Integration of the trained model with the FastAPI backend.
9. Display of diagnosis results through the React dashboard.

## Model Performance

The selected configuration is **XGBoost with SMOTE**.

| Metric | Test Result |
|---|---:|
| Accuracy | 98.66% |
| Macro F1-Score | 0.8747 |

The model was evaluated on a held-out test set. Accuracy and Macro F1-score are both reported to provide a more informative assessment in the presence of class imbalance.

## System Architecture

The application follows this workflow:

**Historical Pump Dataset → Data Preprocessing → Feature Engineering → Machine Learning Model → FastAPI Backend → React Dashboard → Fault Diagnosis**

## Project Structure

```text
intellitwin-fault-diagnosis/
│
├── backend/
│   └── main.py
│
├── data/
│
├── frontend/
│   ├── src/
│   ├── index.html
│   ├── package.json
│   └── vite.config.js
│
├── notebooks/
├── results/
├── scripts/
├── src/
│   └── faultdiag/
├── tests/
│
├── requirements.txt
├── setup_intellitwin.ps1
├── pytest.ini
└── README.md
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/aarnaarao/intellitwin-fault-diagnosis.git
cd intellitwin-fault-diagnosis
```

### 2. Set up the Python environment

```bash
python -m venv .venv
```

Activate the environment on Windows:

```powershell
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### 3. Set up the frontend

```bash
cd frontend
npm install
```

Start the frontend development server:

```bash
npm run dev
```

### 4. Run the backend

Refer to the backend configuration and setup instructions for the appropriate FastAPI startup command.

## Testing

The project includes tests covering:

- Dataset processing
- Feature engineering
- Class imbalance handling
- Data splitting

Run the tests using:

```bash
pytest
```

## Current Scope and Future Enhancements

The current prototype works with historical pump readings and provides machine fault classification and fleet monitoring.

Planned enhancements include:

- Real-time industrial sensor integration.
- Early fault prediction.
- Calibrated failure probability estimation.
- Expanded digital twin capabilities.
- Further improvements to dashboard usability and visual design.

## Author

**Aarna Rao**  
B.Tech – Computer Science and Engineering  
REVA University

## Project Guide

**Dr. Sindhu P. Menon**

---

*IntelliTwin – Intelligent Machine Fault Diagnosis and Failure Prediction.*
