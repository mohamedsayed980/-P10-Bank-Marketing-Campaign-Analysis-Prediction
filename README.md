# 🏦 P10 — Bank Marketing Campaign Analysis & Prediction
**M3 · ML Engine Portfolio · Project 10 of 12**

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit)](https://streamlit.io)
[![Dataset](https://img.shields.io/badge/Source-UCI_Repository-0052CC)](https://archive.ics.uci.edu/dataset/222/bank+marketing)

---

## 📌 Project Overview

End-to-end marketing analytics and ML prediction on **45,211 Portuguese bank clients** contacted during telephone marketing campaigns (2008–2013). The goal is to predict which clients will subscribe to a term deposit and optimise campaign ROI.

**Core Questions:**
- Which client segments have the highest subscription probability?
- How many calls are too many? Where does conversion drop off?
- Does contact method (cellular vs telephone) significantly affect outcomes?
- What is the true ROI of the campaign by segment?

---

## ⚠️ Critical Note — Leakage Column Dropped

```
'duration' column DROPPED — it records the call duration in seconds.
This is ONLY known AFTER the call ends → using it would leak future information.
A realistic predictive model must decide WHO to call BEFORE the call happens.
```

---

## 📊 Dataset

| Property | Value |
|----------|-------|
| Source | UCI Machine Learning Repository |
| File | bank-full.csv |
| Records | 45,211 clients |
| Original Features | 17 |
| After Engineering | 33 |
| Period | 2008–2013 · Portuguese bank |
| Separator | Semicolon (;) — European format |

---

## 🎯 Targets

| Type | Column | Description |
|------|--------|-------------|
| **Regression** | `balance` | Avg yearly account balance (€) |
| **Classification** | `y_binary` | 1 if subscribed, 0 if not |

**Balance:** 11.7% subscribed / 88.3% not → **SEVERE IMBALANCE**

`class_weight='balanced'` **MANDATORY** on ALL classifiers

Evaluate with **F1, Recall, ROC-AUC** — NOT accuracy

---

## ⚙️ Feature Engineering

| Feature | Source | Purpose |
|---------|--------|---------|
| `y_binary` | y == 'yes' | Classification target (0/1) |
| `age_group` | age cut into 6 bands | Non-linear age effect |
| `balance_category` | balance cut into 5 bins | Wealth segmentation |
| `was_contacted_before` | pdays != -1 | Warm lead flag |
| `high_campaign_effort` | campaign > 3 | Resistance signal |
| `prev_success` | poutcome == 'success' | Strongest predictor |
| `month_num` | month mapped 1–12 | Ordered month for correlation |
| `season` | month_num binned | Seasonal subscription patterns |
| `has_debt` | housing='yes' OR loan='yes' | Debt burden flag |
| `*_enc` | LabelEncoder on categoricals | ML-ready features |

---

## 📊 EDA Dashboard — 13 Tabs

| Tab | Title | Highlight |
|-----|-------|-----------|
| 1 | Data Overview | Shape, types, stats, dictionary |
| 2 | Subscription Analysis ★ | Rate by job, education, previous outcome |
| 3 | Campaign Funnel ★ | Calls → contacts → subscriptions |
| 4 | Balance Analysis ★ | Account balance vs subscription rate |
| 5 | Contact Analysis ★ | Method, month, day patterns |
| 6 | Demographics | Age group, marital, education |
| 7 | Multicollinearity | VIF analysis |
| 8 | Correlation | Heatmap + top predictors |
| 9 | Business KPIs ★ | Cost per subscription · ROI by segment |
| 10 | Category Deep-Dive ★ | Job × contact × balance heatmaps |
| 11 | Statistical Tests ★ | T1–T4: cellular vs telephone, balance analysis |
| 12 | Feature Engineering | Engineered flags + distributions |
| 13 | Insights & Report | Findings + recommendations + download |

---

## 🤖 ML Models — 5 Tabs

| Tab | Content |
|-----|---------|
| 1 | Training — 6 Reg + 6 Clf · individual buttons |
| 2 | Regression Results — R², MAE, RMSE · predict balance |
| 3 | Classification Results — F1, Precision, Recall, ROC-AUC |
| 4 | Feature Importance — top subscription predictors |
| 5 | Interactive Predict — subscription probability scorer |

**Regression (6):** Linear · Ridge · Lasso · Decision Tree · Random Forest · Gradient Boosting

**Classification (6):** Logistic Regression · Decision Tree · Random Forest · Gradient Boosting · SVM (Linear) · KNN

---

## 🔑 Key Findings

**1. Previous Campaign Success = #1 Predictor**
Clients who subscribed in a previous campaign are 3–4× more likely to subscribe again. Warm leads are the highest ROI target.

**2. Stop at 3 Calls**
Subscription rate drops sharply after 3 campaign calls. Pursuing resistant clients wastes budget and damages brand perception.

**3. Cellular Outperforms Telephone**
Cellular contact converts significantly better in every demographic segment — reallocate telephone budget to cellular.

**4. Best Months: March, September, October, December**
May has the most calls but lowest conversion — mass campaigns have poor targeting.

**5. Balance Drives Subscription**
Higher account balance = more financial capacity = higher subscription rate. Negative balance clients should be excluded.

**6. Student and Retired Segments = Highest ROI**
Despite small volume, these segments convert at 2–3× the average rate.

---

## 💡 Recommendations

| Priority | Action |
|----------|--------|
| 🎯 High | Target previous subscribers first — 3-4× conversion rate |
| 📱 High | Switch all outreach to cellular — telephone underperforms everywhere |
| 💰 High | Focus on clients with balance > €1,000 |
| 📅 High | Campaign timing: March, September, October, December |
| ⏱ Medium | Stop at 3 calls — redirect budget from resistant clients |
| 👴 Medium | Dedicated campaign for retired clients — highest ROI demographic |

---

## 🗂 Project Structure

```
📁 Repo_10_Bank_Marketing/
├── Home.py
├── M3_logo.png
├── requirements.txt
├── README.md
├── data/
│   └── bank_clean.csv             ← from P10_clean_data.py (Jupyter)
└── pages/
    ├── EDA_dashboard.py            ← 13-tab analysis
    └── ML_Models.py                ← 5-tab ML engine
```

---

## 🚀 How to Run

```bash
git clone https://github.com/YourUsername/Repo_10_Bank_Marketing.git
cd Repo_10_Bank_Marketing

pip install -r requirements.txt

# Step 1: Generate clean dataset in Jupyter
# Run P10_clean_data.py → saves bank_clean.csv
# Copy bank_clean.csv to data/ folder (do NOT open in Excel)

# Step 2: Launch app
streamlit run Home.py
```

> ⚠️ **Important:** Copy CSV directly via File Explorer — never open in Excel before copying. Excel corrupts CSV separators and encoding.

---

## 🛠 Tech Stack

`Python 3.11` · `Streamlit` · `Pandas` · `NumPy` · `Matplotlib` · `Seaborn` · `Plotly` · `Scikit-learn` · `SciPy` · `Statsmodels` · `Psutil`

---

**Mohamed · M3 · ML Engine Portfolio — 12 End-to-End Data Science Projects**
