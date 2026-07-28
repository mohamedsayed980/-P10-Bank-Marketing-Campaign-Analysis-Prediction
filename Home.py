"""
Repo_10_Bank_Marketing — Home.py
Author : Mohamed · M3
"""
import pathlib
import streamlit as st

st.set_page_config(page_title="Bank Marketing · M3", page_icon="🏦", layout="wide")
LOGO = pathlib.Path(__file__).parent / "M3_logo.png"

with st.sidebar:
    if LOGO.exists():
        st.image(str(LOGO), width=70)
    st.markdown("### 🏦 Bank Marketing")
    st.markdown("M3 · ML Engine · P10")
    st.divider()
    st.markdown("**Navigate:**")
    st.markdown("📊 EDA Dashboard → 13 tabs")
    st.markdown("🤖 ML Models     → 5 tabs")

st.markdown("""
<style>
[data-testid="stSidebar"]{background:#0f1923;}
[data-testid="stSidebar"] *{color:#e0e8f0 !important;}
.main{background:#f4f7fb;}
.hero{background:linear-gradient(135deg,#1a237e,#2e7d32);
      padding:48px 40px;border-radius:14px;margin-bottom:28px;}
.hero h1{color:#ffffff !important;font-size:2.4rem;font-weight:800;margin:0 0 8px 0;}
.hero p{color:#c8e6c9 !important;font-size:1.08rem;margin:0;}
.card{background:#ffffff;border-radius:10px;padding:22px 24px;
      box-shadow:0 2px 12px rgba(0,0,0,0.08);border-top:4px solid #1565c0;}
.card h3{color:#1565c0 !important;margin:0 0 8px 0;font-size:1.05rem;}
.card p{color:#37474f !important;font-size:0.92rem;margin:0;line-height:1.6;}
.stat-card{background:#ffffff;border-radius:10px;padding:18px;text-align:center;
           box-shadow:0 2px 10px rgba(0,0,0,0.07);border-bottom:3px solid #1565c0;}
.stat-num{font-size:1.9rem;font-weight:800;color:#1565c0 !important;}
.stat-lbl{font-size:0.82rem;color:#546e7a !important;margin-top:4px;}
</style>""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <h1>🏦 Bank Marketing Campaign Analysis</h1>
  <p>End-to-end ML pipeline · 45,211 clients · UCI Bank Marketing · M3 Portfolio · Project 10 of 12</p>
</div>""", unsafe_allow_html=True)

c1,c2,c3,c4,c5 = st.columns(5)
for col, (num, lbl) in zip([c1,c2,c3,c4,c5],[
    ("45,211","Clients"), ("11.7%","Subscribe Rate"),
    ("17","Features"), ("13","EDA Tabs"), ("12","ML Models")]):
    col.markdown(f"""<div class="stat-card">
      <div class="stat-num">{num}</div>
      <div class="stat-lbl">{lbl}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("### 📌 About This Project")
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("""<div class="card"><h3>🎯 Objective</h3>
    <p>Predict which bank clients will subscribe to a term deposit
    following a telephone marketing campaign. Optimise campaign
    targeting to maximise subscription rate and ROI.</p>
    </div>""", unsafe_allow_html=True)
with c2:
    st.markdown("""<div class="card"><h3>📊 Dataset</h3>
    <p>UCI Bank Marketing · Portuguese bank · 2008–2013 ·
    45,211 clients · 17 original features. Engineered:
    age_group, balance_category, was_contacted_before,
    prev_success, has_debt, season.</p>
    </div>""", unsafe_allow_html=True)
with c3:
    st.markdown("""<div class="card"><h3>🔑 Key Signals</h3>
    <p>Previous campaign outcome · Contact method ·
    Account balance · Age group · Month of contact ·
    Number of campaign calls · Previous contact days.</p>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
col4, col5 = st.columns(2)
with col4:
    st.markdown("### 📈 EDA Dashboard — 13 Tabs")
    for num, name, desc in [
        ("1","Data Overview","Shape, types, stats, dictionary"),
        ("2","Subscription Analysis ★","Overall rate + by job, education, marital"),
        ("3","Campaign Funnel ★","Contact → follow-up → subscribe conversion rates"),
        ("4","Balance Analysis ★","Account balance vs subscription rate"),
        ("5","Contact Analysis ★","Method, day, month patterns"),
        ("6","Demographics","Age group, job, marital, education"),
        ("7","Multicollinearity","VIF analysis"),
        ("8","Correlation","Heatmap + top subscription predictors"),
        ("9","Business KPIs ★","Cost per subscription · ROI by segment"),
        ("10","Category Deep-Dive ★","Job × contact method × balance cross-analysis"),
        ("11","Statistical Tests ★","T1-T4: cellular vs telephone · balance by outcome"),
        ("12","Feature Engineering","Engineered flags + distributions"),
        ("13","Insights & Report","Findings + recommendations + download"),
    ]:
        st.markdown(f"**Tab {num} · {name}** — {desc}")

with col5:
    st.markdown("### 🤖 ML Models — 5 Tabs")
    for num, name, desc in [
        ("1","Model Training","6 Reg + 6 Clf · individual buttons"),
        ("2","Regression Results","R², MAE, RMSE · predict account balance"),
        ("3","Classification Results","F1, Recall, ROC-AUC · predict subscription"),
        ("4","Feature Importance","Top subscription predictors"),
        ("5","Predict","Interactive subscription probability scorer"),
    ]:
        st.markdown(f"**Tab {num} · {name}** — {desc}")

    st.markdown("<br>", unsafe_allow_html=True)
    st.warning("**Subscribe Rate: 11.7%** — severe imbalance.\n\n"
               "All classifiers use `class_weight='balanced'`.\n\n"
               "⚠️ `duration` dropped — post-call leakage.\n\n"
               "Evaluate with **F1, Recall, ROC-AUC**.")

st.markdown("---")
st.markdown("<p style='text-align:center;color:#90a4ae;font-size:0.85rem;'>"
            "Mohamed · M3 · ML Engine Portfolio · Project 10 of 12 · Bank Marketing</p>",
            unsafe_allow_html=True)
