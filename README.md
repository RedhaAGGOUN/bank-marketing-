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

💻 Running Locally
Clone the repo:

bash
Copy
Edit
git clone https://github.com/yourusername/bank-marketing-predictor.git
Install dependencies:

bash
Copy
Edit
conda activate bankmarketing
pip install -r requirements.txt
Launch the Streamlit app:

bash
Copy
Edit
streamlit run Prediction_app_improved.py
Visit http://localhost:8501.

🎯 Streamlit Application Overview
✅ Single Client Analysis

Enter customer features

See their subscription probability

Get suggestions for campaign tweaks if their probability is low

✅ Strategic Campaign Planner

Segment customers into high-potential vs. low-potential

Suggest optimal campaign settings (month, duration, contact type)

Provide data-driven recommendations for maximizing campaign ROI

🔎 Dataset
Source: UCI Bank Marketing Data Set

Size: 45,211 rows, 17 columns

Includes:

bank client data (age, job, marital, education, default, housing, loan, balance)

contact data (contact type, day, month, duration)

campaign data (campaign, pdays, previous, poutcome)

target: y (term deposit subscription)

🌟 Next Steps
Deploy on a cloud server (AWS, Streamlit Cloud)

Integrate with CRM systems for real-time predictions

Build an uplift modeling module

Automate retraining pipelines

👨‍💻 Author
Redha  Aggoun
Feel free to reach out for collaboration or more advanced ML projects.

📄 License
This project is for educational demonstration. For commercial use, please contact the author.

Enjoy predicting & strategizing! 🎯

