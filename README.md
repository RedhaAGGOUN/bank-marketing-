# 🏦 Strategic Bank Marketing Term Deposit Predictor 🎯

> *“Transforming raw marketing data into actionable strategies with machine learning and data science.”*

---

## 📌 Project Overview

This project leverages a real-world Portuguese banking dataset to build a predictive model for **term deposit subscription** outcomes. It uses CRISP-DM methodology with rigorous data understanding, preprocessing, feature engineering, model selection, and deployment through a beautiful Streamlit web application.

By identifying which clients are more likely to subscribe to a term deposit, the bank can:
- reduce unnecessary calls,
- focus efforts on high-probability prospects,
- design smarter campaigns with optimized parameters,
- and maximize marketing ROI.

---

## 🚀 Features

✅ **Exploratory Data Analysis (EDA)**  
✅ Robust outlier detection (IQR rule) and trimming  
✅ Binary encoding of key features  
✅ Feature engineering (e.g., `was_contacted_before`)  
✅ Advanced modeling with PyCaret classification  
✅ Best model exported with `joblib`  
✅ Full-featured **Streamlit** interface with:
  - Single-client analysis
  - Strategic campaign planner
  - Minimal change recommender
  - Attractive, modern UI
✅ Modular, reproducible, documented code

---

## 📂 Project Structure

```plaintext
.
├── bank_marketing_model.pkl          # Trained PyCaret model (joblib)
├── bank_subscription_model.pkl       # Alternative trained model
├── bankmarketing_lightgbm.pkl        # LightGBM checkpoint from PyCaret
├── bank-full.csv                     # Raw data
├── Clean_bank_marketing.ipynb        # Full exploratory + modeling notebook
├── bank_marketing.ipynb              # Another development notebook
├── Prediction_app.py                 # Streamlit application (basic version)
├── Prediction_app_improved.py        # Streamlit application (advanced version)
├── logs.log                          # session logs
└── README.md                         # you are here
🛠️ Tech Stack
Python (pandas, scikit-learn, pycaret, plotly, joblib)

Streamlit (user interface)

Jupyter Notebook for experimentation

VS Code / Anaconda for environment control

📈 Results & Insights
✅ Achieved >90% accuracy with LightGBM
✅ Discovered that call duration, previous campaign outcome, and day of contact strongly influence success
✅ Built actionable marketing personas
✅ Created a minimum-changes recommender to convert hesitant clients
✅ Developed a strategic planner to recommend whom to contact and when


