"""
Repo_10_Bank_Marketing — EDA_dashboard.py  (13 Tabs)
Author : Mohamed · M3
Dataset: UCI Bank Marketing · 45,211 clients · Portuguese bank 2008-2013
"""
import pathlib, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
import streamlit as st

warnings.filterwarnings("ignore")
S = st.session_state

st.set_page_config(page_title="EDA · Bank Marketing · M3",
                   page_icon="🏦", layout="wide")

LOGO = pathlib.Path(__file__).parent.parent / "M3_logo.png"
DATA = pathlib.Path(__file__).parent.parent / "data" / "bank_clean.csv"

# ── SIDEBAR ──────────────────────────────────────────────────
with st.sidebar:
    if LOGO.exists():
        st.image(str(LOGO), width=70)
    st.markdown("### 🏦 EDA Dashboard")
    st.markdown("Bank Marketing · 13 Tabs")
    st.divider()
    st.markdown("### 📂 Dataset")
    st.success("✅ bank_clean.csv")
    st.caption("Loaded from data/ folder")

CLR = {"primary":"#1565c0","success":"#2e7d32","warning":"#e65100",
       "danger":"#c62828","teal":"#00695c","light":"#e3f2fd","dark":"#1a237e",
       "purple":"#6a1b9a","amber":"#f57f17","grey":"#546e7a","white":"#ffffff"}

st.markdown("""
<style>
[data-testid="stSidebar"]{background:#0f1923;}
[data-testid="stSidebar"] *{color:#e0e8f0 !important;}
[data-testid="stSidebar"] [data-testid="stFileUploader"]{background:#1a2633;border:1.5px dashed #4a7fa5;border-radius:8px;padding:6px;}
[data-testid="stSidebar"] [data-testid="stFileUploader"] *{color:#e0e8f0 !important;}
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"]{background:#1a2633 !important;border:none !important;}
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzoneInstructions"] *{color:#a0bcd4 !important;}
[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"]{background:#1565c0 !important;color:#ffffff !important;border:none !important;border-radius:6px !important;}
.main{background:#f4f7fb;}
div[data-testid="metric-container"]{background:#e3f2fd;border-left:4px solid #1565c0;border-radius:6px;padding:10px 14px;}
.sec-header{background:linear-gradient(90deg,#1565c0,#2e7d32);color:#ffffff !important;
  padding:10px 18px;border-radius:8px;font-size:1.1rem;font-weight:700;margin-bottom:16px;}
.insight-box{background:#e8f5e9;border-left:4px solid #2e7d32;padding:12px 16px;border-radius:0 6px 6px 0;margin:8px 0;}
.insight-box p{color:#1b3a1f !important;margin:0;font-size:0.93rem;line-height:1.6;}
.warn-box{background:#fff3e0;border-left:4px solid #e65100;padding:12px 16px;border-radius:0 6px 6px 0;margin:8px 0;}
.warn-box p{color:#4a2000 !important;margin:0;font-size:0.93rem;line-height:1.6;}
.info-box{background:#e3f2fd;border-left:4px solid #1565c0;padding:12px 16px;border-radius:0 6px 6px 0;margin:8px 0;}
.info-box p{color:#0d2a4a !important;margin:0;font-size:0.93rem;line-height:1.6;}
</style>""", unsafe_allow_html=True)

def sec(t): st.markdown(f'<div class="sec-header">{t}</div>', unsafe_allow_html=True)
def insight(t): st.markdown(f'<div class="insight-box"><p>✅ {t}</p></div>', unsafe_allow_html=True)
def warn(t):    st.markdown(f'<div class="warn-box"><p>⚠️ {t}</p></div>', unsafe_allow_html=True)
def info(t):    st.markdown(f'<div class="info-box"><p>ℹ️ {t}</p></div>', unsafe_allow_html=True)

# ── LOAD — direct from data/ using os.path (most reliable) ──
import os as _os

_data_path = _os.path.join(
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
    "data", "bank_clean.csv"
)

if not _os.path.exists(_data_path):
    st.error(f"❌ File not found: {_data_path}")
    st.info("Place bank_clean.csv in the data/ folder then restart Streamlit.")
    st.stop()

try:
    # Auto-detect separator — handles both comma and semicolon saved files
    with open(_data_path, 'r', encoding='utf-8') as _f:
        _first_line = _f.readline()
    _sep = ";" if _first_line.count(";") > _first_line.count(",") else ","
    df = pd.read_csv(_data_path, sep=_sep, decimal="." if _sep=="," else ",")
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    df.columns = df.columns.str.strip()
except Exception as e:
    st.error(f"❌ Load error: {e}")
    st.stop()

if df.empty:
    st.warning("⚠️ Dataset is empty.")
    st.stop()



# ── COLUMN GUARD ─────────────────────────────────────────────
# If raw bank-full.csv was uploaded instead of bank_clean.csv,
# the engineered columns won't exist. Detect and guide the user.


S["df_work"] = df

TARGET  = "y_binary"
REG_T   = "balance"
# ── DEBUG: show what was actually loaded ────────────────────
st.sidebar.caption(f"Rows: {df.shape[0]:,} · Cols: {df.shape[1]}")
st.sidebar.caption(f"Has y_binary: {'y_binary' in df.columns}")

# Safe check — if wrong file stop with clear message
if "y_binary" not in df.columns:
    st.error("❌ Wrong file in data/ folder — this is the RAW bank-full.csv")
    st.warning(f"File loaded from: {_data_path}")
    st.info("In Jupyter, run P10_clean_data.py → it saves bank_clean.csv → REPLACE the file in data/ with this new one")
    st.code(f"Columns found ({df.shape[1]}): {list(df.columns)}")
    st.stop()

sub_rate = df[TARGET].mean() * 100
df_sub  = df[df[TARGET] == 1]
df_nos  = df[df[TARGET] == 0]

NUM_COLS = [c for c in ["age","balance","campaign","pdays","previous",
                         "month_num","day"] if c in df.columns]
CAT_COLS = [c for c in ["job","marital","education","default",
                         "housing","loan","contact","month","poutcome"] if c in df.columns]
ENG_COLS = [c for c in ["age_group","balance_category","was_contacted_before",
                         "high_campaign_effort","prev_success","has_debt","season"] if c in df.columns]

# ── TABS ─────────────────────────────────────────────────────
tabs = st.tabs([
    "1 · Data Overview",
    "2 · Subscription Analysis ★",
    "3 · Campaign Funnel ★",
    "4 · Balance Analysis ★",
    "5 · Contact Analysis ★",
    "6 · Demographics",
    "7 · Multicollinearity",
    "8 · Correlation",
    "9 · Business KPIs ★",
    "10 · Category Deep-Dive ★",
    "11 · Statistical Tests ★",
    "12 · Feature Engineering",
    "13 · Insights & Report",
])

# ══════════════════════════════════════════════════════════════
# TAB 1 — DATA OVERVIEW
# ══════════════════════════════════════════════════════════════
with tabs[0]:
    sec("📋 Tab 1 — Data Overview")
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Total Clients",   f"{len(df):,}")
    c2.metric("Subscribed",      f"{len(df_sub):,}")
    c3.metric("Not Subscribed",  f"{len(df_nos):,}")
    c4.metric("Subscribe Rate",  f"{sub_rate:.1f}%")
    c5.metric("Features",        f"{df.shape[1]}")

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        sec("📄 First 10 Rows")
        st.dataframe(df.head(10), use_container_width=True)
    with col2:
        sec("📐 Column Info")
        info_df = pd.DataFrame({
            "Column": df.columns,
            "Dtype":  df.dtypes.astype(str).values,
            "Nulls":  df.isnull().sum().values,
        })
        st.dataframe(info_df, use_container_width=True)

    st.markdown("---")
    sec("📊 Descriptive Statistics")
    st.dataframe(df[NUM_COLS].describe().round(3), use_container_width=True)

    st.markdown("---")
    sec("🗂 Data Dictionary")
    dd = pd.DataFrame({
        "Column":      ["age","job","marital","education","default","balance",
                        "housing","loan","contact","day","month","campaign",
                        "pdays","previous","poutcome","y_binary",
                        "age_group","balance_category","was_contacted_before",
                        "high_campaign_effort","prev_success","has_debt","season"],
        "Type":        ["Numeric","Categorical","Categorical","Categorical","Binary","Numeric",
                        "Binary","Binary","Categorical","Numeric","Categorical","Numeric",
                        "Numeric","Numeric","Categorical","Target",
                        "Engineered","Engineered","Engineered",
                        "Engineered","Engineered","Engineered","Engineered"],
        "Description": [
            "Client age in years",
            "Type of job (admin, blue-collar, entrepreneur, etc.)",
            "Marital status (married/single/divorced)",
            "Education level (primary/secondary/tertiary/unknown)",
            "Has credit in default? (1=yes 0=no)",
            "Average yearly account balance in euros — REG TARGET",
            "Has housing loan? (yes/no)",
            "Has personal loan? (yes/no)",
            "Contact communication type (cellular/telephone/unknown)",
            "Last contact day of month",
            "Last contact month of year",
            "Number of contacts during this campaign",
            "Days since last contact from previous campaign (-1=never)",
            "Number of contacts before this campaign",
            "Outcome of previous campaign (success/failure/other/unknown)",
            "Did client subscribe? 1=yes 0=no — CLF TARGET",
            "Age bin: 18-25 / 26-35 / 36-45 / 46-55 / 56-65 / 65+",
            "Balance bin: Negative / Low / Medium / High / Very High",
            "1 if pdays != -1 (was contacted in previous campaign)",
            "1 if campaign > 3 calls (high effort, lower conversion)",
            "1 if poutcome == 'success'",
            "1 if housing='yes' OR loan='yes'",
            "Season of last contact month",
        ]
    })
    st.dataframe(dd, use_container_width=True)
    warn(f"Subscribe rate {sub_rate:.1f}% — severe imbalance → class_weight='balanced' mandatory.")
    warn("'duration' column DROPPED — post-call leakage (only known after call ends).")

# ══════════════════════════════════════════════════════════════
# TAB 2 — SUBSCRIPTION ANALYSIS ★
# ══════════════════════════════════════════════════════════════
with tabs[1]:
    sec("🎯 Tab 2 — Subscription Analysis ★")
    info("Where do subscriptions concentrate? Previous success is the strongest signal.")

    col1, col2 = st.columns(2)
    with col1:
        sec("📊 Overall Balance")
        bal = pd.DataFrame({"Label":["Not Subscribed","Subscribed"],
                            "Count":[len(df_nos), len(df_sub)]})
        bal["Pct"] = (bal["Count"]/len(df)*100).round(1)
        fig = px.bar(bal, x="Label", y="Count", color="Label",
                     color_discrete_map={"Not Subscribed":CLR["grey"],"Subscribed":CLR["success"]},
                     text=bal["Pct"].apply(lambda x: f"{x}%"),
                     title="Subscribed vs Not Subscribed")
        fig.update_traces(textposition="outside")
        fig.update_layout(height=370, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        sec("📊 Subscribe Rate by Previous Outcome")
        if "poutcome" in df.columns:
            po = df.groupby("poutcome")[TARGET].agg(Total="count",Sub="sum").reset_index()
            po["Rate%"] = (po["Sub"]/po["Total"]*100).round(2)
            po = po.sort_values("Rate%", ascending=False)
            fig2 = px.bar(po, x="poutcome", y="Rate%",
                          color="Rate%",
                          color_continuous_scale=["#c62828","#e65100","#2e7d32"],
                          title="Subscribe Rate % by Previous Outcome",
                          text=po["Rate%"].apply(lambda x: f"{x:.1f}%"))
            fig2.add_hline(y=sub_rate, line_dash="dash", line_color="blue",
                           annotation_text=f"Avg {sub_rate:.1f}%")
            fig2.update_traces(textposition="outside")
            fig2.update_layout(height=370)
            st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    col3, col4 = st.columns(2)
    with col3:
        sec("📊 Subscribe Rate by Job")
        jr = df.groupby("job")[TARGET].agg(Total="count",Sub="sum").reset_index()
        jr["Rate%"] = (jr["Sub"]/jr["Total"]*100).round(2)
        jr = jr.sort_values("Rate%", ascending=False)
        fig3 = px.bar(jr, x="job", y="Rate%",
                      color="Rate%",
                      color_continuous_scale=["#c62828","#e65100","#2e7d32"],
                      title="Subscribe Rate % by Job",
                      text=jr["Rate%"].apply(lambda x: f"{x:.1f}%"))
        fig3.add_hline(y=sub_rate, line_dash="dash", line_color="blue")
        fig3.update_traces(textposition="outside")
        fig3.update_layout(height=400, xaxis_tickangle=-30)
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        sec("📊 Subscribe Rate by Education")
        er = df.groupby("education")[TARGET].agg(Total="count",Sub="sum").reset_index()
        er["Rate%"] = (er["Sub"]/er["Total"]*100).round(2)
        er = er.sort_values("Rate%", ascending=False)
        fig4 = px.bar(er, x="education", y="Rate%",
                      color="Rate%",
                      color_continuous_scale=["#c62828","#e65100","#2e7d32"],
                      title="Subscribe Rate % by Education",
                      text=er["Rate%"].apply(lambda x: f"{x:.1f}%"))
        fig4.add_hline(y=sub_rate, line_dash="dash", line_color="blue")
        fig4.update_traces(textposition="outside")
        fig4.update_layout(height=370)
        st.plotly_chart(fig4, use_container_width=True)

    insight("Previous campaign SUCCESS is the #1 predictor — clients who subscribed before are 3-4× more likely to subscribe again.")
    insight("Students and retired clients show highest subscription rates — more time + financial planning motivation.")
    warn("Blue-collar workers show lowest subscription rate — less financial product engagement.")

# ══════════════════════════════════════════════════════════════
# TAB 3 — CAMPAIGN FUNNEL ★
# ══════════════════════════════════════════════════════════════
with tabs[2]:
    sec("🔄 Tab 3 — Campaign Funnel ★")
    info("How many calls → contacts → subscriptions? The funnel shows conversion efficiency.")

    col1, col2 = st.columns(2)
    with col1:
        sec("📊 Subscribe Rate by Number of Calls (campaign)")
        camp = df.groupby("campaign")[TARGET].agg(Total="count",Sub="sum").reset_index()
        camp["Rate%"] = (camp["Sub"]/camp["Total"]*100).round(2)
        camp = camp[camp["campaign"] <= 15]  # cap at 15 for clarity
        fig = px.bar(camp, x="campaign", y="Rate%",
                     color="Rate%",
                     color_continuous_scale=["#2e7d32","#e65100","#c62828"],
                     title="Subscribe Rate % by Number of Calls",
                     text=camp["Rate%"].apply(lambda x: f"{x:.1f}%"))
        fig.add_hline(y=sub_rate, line_dash="dash", line_color="blue",
                      annotation_text=f"Avg {sub_rate:.1f}%")
        fig.update_traces(textposition="outside")
        fig.update_layout(height=380, xaxis_title="Number of Campaign Calls")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        sec("📊 Subscribe Rate — Previous vs New Contacts")
        if "was_contacted_before" in df.columns:
            wc = df.groupby("was_contacted_before")[TARGET].agg(
                Total="count",Sub="sum").reset_index()
            wc["Label"] = wc["was_contacted_before"].map(
                {0:"First Time Contact", 1:"Previously Contacted"})
            wc["Rate%"] = (wc["Sub"]/wc["Total"]*100).round(2)
            fig2 = px.bar(wc, x="Label", y="Rate%", color="Label",
                          color_discrete_map={"First Time Contact":CLR["grey"],
                                              "Previously Contacted":CLR["success"]},
                          title="Subscribe Rate: First Time vs Previously Contacted",
                          text=wc["Rate%"].apply(lambda x: f"{x:.1f}%"))
            fig2.update_traces(textposition="outside")
            fig2.update_layout(height=380, showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    sec("📊 Funnel — Campaign Effort vs Conversion")
    if "high_campaign_effort" in df.columns:
        hce = df.groupby("high_campaign_effort")[TARGET].agg(
            Total="count",Sub="sum").reset_index()
        hce["Label"] = hce["high_campaign_effort"].map(
            {0:"≤3 Calls (Efficient)", 1:">3 Calls (High Effort)"})
        hce["Rate%"] = (hce["Sub"]/hce["Total"]*100).round(2)
        col3, col4 = st.columns(2)
        with col3:
            st.dataframe(hce[["Label","Total","Sub","Rate%"]], use_container_width=True)
        with col4:
            fig3 = px.bar(hce, x="Label", y="Rate%", color="Label",
                          color_discrete_map={"≤3 Calls (Efficient)":CLR["success"],
                                              ">3 Calls (High Effort)":CLR["danger"]},
                          title="Subscribe Rate: Efficient vs High Effort",
                          text=hce["Rate%"].apply(lambda x: f"{x:.1f}%"))
            fig3.update_traces(textposition="outside")
            fig3.update_layout(height=320, showlegend=False)
            st.plotly_chart(fig3, use_container_width=True)

    insight("First call converts best — subscription rate drops sharply after 3+ calls.")
    insight("Previously contacted clients convert significantly better — warm leads are valuable.")
    warn("High campaign effort (>3 calls) actually LOWERS conversion — stop pursuing resistant clients.")

# ══════════════════════════════════════════════════════════════
# TAB 4 — BALANCE ANALYSIS ★
# ══════════════════════════════════════════════════════════════
with tabs[3]:
    sec("💰 Tab 4 — Balance Analysis ★")
    info("Account balance is the regression target AND a strong subscription predictor.")

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Subscribed — Avg Balance",     f"€{df_sub[REG_T].mean():,.0f}")
    c2.metric("Not Subscribed — Avg Balance", f"€{df_nos[REG_T].mean():,.0f}")
    c3.metric("Overall Median",               f"€{df[REG_T].median():,.0f}")
    c4.metric("Negative Balance %",           f"{(df[REG_T]<0).mean()*100:.1f}%")

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        sec("📊 Balance Distribution — Subscribed vs Not")
        fig, ax = plt.subplots(figsize=(8,4))
        df_nos[REG_T].clip(-500, 5000).hist(bins=50, ax=ax, alpha=0.6,
            color=CLR["grey"], density=True, label="Not Subscribed")
        df_sub[REG_T].clip(-500, 5000).hist(bins=50, ax=ax, alpha=0.7,
            color=CLR["success"], density=True, label="Subscribed")
        ax.axvline(df_nos[REG_T].mean(), color=CLR["grey"],   lw=2, ls="--")
        ax.axvline(df_sub[REG_T].mean(), color=CLR["success"],lw=2, ls="--")
        ax.set_xlabel("Balance (€, clipped at -500 to 5000)")
        ax.set_ylabel("Density")
        ax.set_title("Balance: Subscribed vs Not Subscribed")
        ax.legend()
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with col2:
        sec("📊 Subscribe Rate by Balance Category")
        if "balance_category" in df.columns:
            bc = df.groupby("balance_category", observed=True)[TARGET].agg(
                Total="count",Sub="sum").reset_index()
            bc["Rate%"] = (bc["Sub"]/bc["Total"]*100).round(2)
            fig2 = px.bar(bc, x="balance_category", y="Rate%",
                          color="Rate%",
                          color_continuous_scale=["#c62828","#e65100","#2e7d32"],
                          title="Subscribe Rate % by Balance Category",
                          text=bc["Rate%"].apply(lambda x: f"{x:.1f}%"))
            fig2.add_hline(y=sub_rate, line_dash="dash", line_color="blue")
            fig2.update_traces(textposition="outside")
            fig2.update_layout(height=380)
            st.plotly_chart(fig2, use_container_width=True)

    insight("Clients with higher balances subscribe more — financial capacity drives investment decisions.")
    warn("Clients with negative balances should be excluded from term deposit campaigns — no capacity to invest.")

# ══════════════════════════════════════════════════════════════
# TAB 5 — CONTACT ANALYSIS ★
# ══════════════════════════════════════════════════════════════
with tabs[4]:
    sec("📞 Tab 5 — Contact Analysis ★")
    info("When and how you contact matters as much as who you contact.")

    col1, col2 = st.columns(2)
    with col1:
        sec("📊 Subscribe Rate by Contact Method")
        if "contact" in df.columns:
            cm = df.groupby("contact")[TARGET].agg(Total="count",Sub="sum").reset_index()
            cm["Rate%"] = (cm["Sub"]/cm["Total"]*100).round(2)
            cm = cm.sort_values("Rate%", ascending=False)
            fig = px.bar(cm, x="contact", y="Rate%",
                         color="Rate%",
                         color_continuous_scale=["#c62828","#e65100","#2e7d32"],
                         title="Subscribe Rate % by Contact Method",
                         text=cm["Rate%"].apply(lambda x: f"{x:.1f}%"))
            fig.add_hline(y=sub_rate, line_dash="dash", line_color="blue")
            fig.update_traces(textposition="outside")
            fig.update_layout(height=370)
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        sec("📊 Subscribe Rate by Month")
        if "month" in df.columns:
            month_order = ['jan','feb','mar','apr','may','jun',
                           'jul','aug','sep','oct','nov','dec']
            mr = df.groupby("month")[TARGET].agg(Total="count",Sub="sum").reset_index()
            mr["Rate%"] = (mr["Sub"]/mr["Total"]*100).round(2)
            mr["month"] = pd.Categorical(mr["month"],
                categories=[m for m in month_order if m in mr["month"].values], ordered=True)
            mr = mr.sort_values("month")
            fig2 = px.bar(mr, x="month", y="Rate%",
                          color="Rate%",
                          color_continuous_scale=["#c62828","#e65100","#2e7d32"],
                          title="Subscribe Rate % by Month",
                          text=mr["Rate%"].apply(lambda x: f"{x:.1f}%"))
            fig2.add_hline(y=sub_rate, line_dash="dash", line_color="blue")
            fig2.update_traces(textposition="outside")
            fig2.update_layout(height=370)
            st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    sec("📊 Subscribe Rate by Day of Month")
    if "day" in df.columns:
        dr = df.groupby("day")[TARGET].agg(Total="count",Sub="sum").reset_index()
        dr["Rate%"] = (dr["Sub"]/dr["Total"]*100).round(2)
        fig3 = px.line(dr, x="day", y="Rate%", markers=True,
                       title="Subscribe Rate % by Day of Month",
                       color_discrete_sequence=[CLR["primary"]])
        fig3.add_hline(y=sub_rate, line_dash="dash", line_color="red",
                       annotation_text=f"Avg {sub_rate:.1f}%")
        fig3.update_layout(height=350)
        st.plotly_chart(fig3, use_container_width=True)

    insight("Cellular contact outperforms telephone — mobile reaches clients directly.")
    insight("March, September, October, December show highest subscription rates.")
    warn("May has the MOST calls but lowest conversion rate — mass campaigns = poor targeting.")

# ══════════════════════════════════════════════════════════════
# TAB 6 — DEMOGRAPHICS
# ══════════════════════════════════════════════════════════════
with tabs[5]:
    sec("👥 Tab 6 — Demographics")

    col1, col2 = st.columns(2)
    with col1:
        sec("📊 Subscribe Rate by Age Group")
        if "age_group" in df.columns:
            ag = df.groupby("age_group", observed=True)[TARGET].agg(
                Total="count",Sub="sum").reset_index()
            ag["Rate%"] = (ag["Sub"]/ag["Total"]*100).round(2)
            fig = px.bar(ag, x="age_group", y="Rate%",
                         color="Rate%",
                         color_continuous_scale=["#c62828","#e65100","#2e7d32"],
                         title="Subscribe Rate % by Age Group",
                         text=ag["Rate%"].apply(lambda x: f"{x:.1f}%"))
            fig.add_hline(y=sub_rate, line_dash="dash", line_color="blue")
            fig.update_traces(textposition="outside")
            fig.update_layout(height=360)
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        sec("📊 Subscribe Rate by Marital Status")
        if "marital" in df.columns:
            ms = df.groupby("marital")[TARGET].agg(Total="count",Sub="sum").reset_index()
            ms["Rate%"] = (ms["Sub"]/ms["Total"]*100).round(2)
            fig2 = px.bar(ms, x="marital", y="Rate%", color="marital",
                          title="Subscribe Rate % by Marital Status",
                          text=ms["Rate%"].apply(lambda x: f"{x:.1f}%"))
            fig2.update_traces(textposition="outside")
            fig2.update_layout(height=360, showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    sec("📊 Age Distribution — Subscribed vs Not")
    fig3, ax = plt.subplots(figsize=(12,4))
    ax.hist(df_nos["age"], bins=40, alpha=0.6,
            color=CLR["grey"], density=True, label="Not Subscribed")
    ax.hist(df_sub["age"], bins=40, alpha=0.7,
            color=CLR["success"], density=True, label="Subscribed")
    ax.axvline(df_nos["age"].mean(), color=CLR["grey"],   lw=2, ls="--")
    ax.axvline(df_sub["age"].mean(), color=CLR["success"],lw=2, ls="--")
    ax.set_xlabel("Age"); ax.set_ylabel("Density")
    ax.set_title("Age Distribution: Subscribed vs Not Subscribed")
    ax.legend()
    plt.tight_layout(); st.pyplot(fig3); plt.close()

    insight("Young (18-25) and retired (65+) have highest subscribe rates — different motivations.")
    insight("Single clients subscribe slightly more than married — fewer financial commitments.")

# ══════════════════════════════════════════════════════════════
# TAB 7 — MULTICOLLINEARITY
# ══════════════════════════════════════════════════════════════
with tabs[6]:
    sec("🔁 Tab 7 — Multicollinearity / VIF")
    info("VIF > 10 = severe multicollinearity. Critical for Logistic Regression accuracy.")

    vif_cols = [c for c in ["age","balance","campaign","pdays","previous",
                              "day","month_num","has_debt","prev_success",
                              "was_contacted_before","high_campaign_effort"]
                if c in df.columns]
    vif_data = df[vif_cols].dropna()
    try:
        vif_df = pd.DataFrame({
            "Feature": vif_cols,
            "VIF": [round(variance_inflation_factor(vif_data.values, i), 2)
                    for i in range(len(vif_cols))]
        }).sort_values("VIF", ascending=False)
        vif_df["Risk"] = vif_df["VIF"].apply(
            lambda v: "🔴 High" if v>10 else "🟡 Medium" if v>5 else "🟢 Low")

        col1, col2 = st.columns([1, 1.5])
        with col1:
            st.dataframe(vif_df, use_container_width=True)
        with col2:
            fig, ax = plt.subplots(figsize=(7, 5))
            colors_vif = [CLR["danger"] if v>10 else CLR["warning"] if v>5
                          else CLR["success"] for v in vif_df["VIF"]]
            ax.barh(vif_df["Feature"], vif_df["VIF"], color=colors_vif)
            ax.axvline(10, color=CLR["danger"],  lw=2, ls="--", label="VIF=10")
            ax.axvline(5,  color=CLR["warning"], lw=1.5, ls=":", label="VIF=5")
            ax.set_xlabel("VIF"); ax.set_title("Multicollinearity Check")
            ax.legend(); plt.tight_layout(); st.pyplot(fig); plt.close()
    except Exception as e:
        warn(f"VIF error: {e}")

    insight("Tree-based models (RF, GB) are immune to multicollinearity — VIF only matters for Logistic Regression.")
    warn("pdays and was_contacted_before are derived from the same source — expect high correlation.")

# ══════════════════════════════════════════════════════════════
# TAB 8 — CORRELATION
# ══════════════════════════════════════════════════════════════
with tabs[7]:
    sec("🔥 Tab 8 — Correlation Analysis")

    enc_cols = [c for c in df.columns if c.endswith("_enc")]
    corr_cols = [c for c in NUM_COLS + enc_cols + [TARGET] if c in df.columns]
    corr = df[corr_cols].corr()

    fig, ax = plt.subplots(figsize=(13, 9))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlBu_r",
                vmin=-1, vmax=1, ax=ax, linewidths=0.5, annot_kws={"size": 8})
    ax.set_title("Correlation Matrix — Bank Marketing Features", fontsize=13, fontweight="bold")
    plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown("---")
    sec("🎯 Top Correlations with y_binary")
    tgt = corr[TARGET].drop(TARGET).sort_values(key=abs, ascending=False).head(12)
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    colors_bar = [CLR["success"] if v > 0 else CLR["danger"] for v in tgt.values]
    ax2.barh(tgt.index, tgt.values, color=colors_bar)
    ax2.axvline(0, color="black", lw=0.8)
    ax2.set_xlabel("Pearson r with y_binary")
    ax2.set_title("Feature Correlation with Subscription", fontsize=12, fontweight="bold")
    for i, (idx, val) in enumerate(tgt.items()):
        ax2.text(val + 0.003 if val >= 0 else val - 0.003, i,
                 f"{val:.3f}", va="center",
                 ha="left" if val >= 0 else "right", fontsize=9)
    plt.tight_layout(); st.pyplot(fig2); plt.close()

    insight("poutcome_enc (previous outcome) shows strongest correlation with subscription.")
    warn("Pearson r misses non-linear effects — use RF/GB feature importance for full picture.")

# ══════════════════════════════════════════════════════════════
# TAB 9 — BUSINESS KPIs ★
# ══════════════════════════════════════════════════════════════
with tabs[8]:
    sec("💼 Tab 9 — Business KPIs ★")
    info("Translate data findings into business metrics: cost per conversion, best ROI segments.")

    # Assume average cost per call = €5, average term deposit value = €2,000
    COST_PER_CALL = 5
    AVG_DEPOSIT   = 2000

    total_calls   = df["campaign"].sum()
    total_subs    = df[TARGET].sum()
    total_cost    = total_calls * COST_PER_CALL
    total_revenue = total_subs * AVG_DEPOSIT
    cost_per_sub  = total_cost / max(total_subs, 1)
    roi           = (total_revenue - total_cost) / max(total_cost, 1) * 100

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total Calls Made",       f"{total_calls:,}")
    c2.metric("Total Subscriptions",    f"{total_subs:,}")
    c3.metric("Cost per Subscription",  f"€{cost_per_sub:,.0f}")
    c4.metric("Campaign ROI",           f"{roi:,.0f}%")

    st.markdown("---")
    sec("📊 ROI by Job Segment")
    job_kpi = df.groupby("job").agg(
        Calls=("campaign","sum"),
        Subs=(TARGET,"sum"),
        Clients=(TARGET,"count")
    ).reset_index()
    job_kpi["Cost"]        = job_kpi["Calls"] * COST_PER_CALL
    job_kpi["Revenue"]     = job_kpi["Subs"] * AVG_DEPOSIT
    job_kpi["ROI%"]        = ((job_kpi["Revenue"] - job_kpi["Cost"]) /
                               job_kpi["Cost"].clip(lower=1) * 100).round(0)
    job_kpi["SubRate%"]    = (job_kpi["Subs"]/job_kpi["Clients"]*100).round(1)
    job_kpi["CostPerSub"]  = (job_kpi["Cost"]/job_kpi["Subs"].clip(lower=1)).round(0)
    job_kpi = job_kpi.sort_values("ROI%", ascending=False)

    col1, col2 = st.columns(2)
    with col1:
        st.dataframe(job_kpi[["job","SubRate%","CostPerSub","ROI%"]].round(0),
                     use_container_width=True)
    with col2:
        fig = px.bar(job_kpi, x="job", y="ROI%",
                     color="ROI%", color_continuous_scale=["#c62828","#e65100","#2e7d32"],
                     title="Campaign ROI % by Job Segment",
                     text=job_kpi["ROI%"].apply(lambda x: f"{x:.0f}%"))
        fig.update_traces(textposition="outside")
        fig.update_layout(height=380, xaxis_tickangle=-30)
        st.plotly_chart(fig, use_container_width=True)

    insight(f"Total campaign ROI: {roi:,.0f}% — term deposits are highly profitable even with low subscribe rates.")
    warn("Students and retired segments deliver highest ROI despite small volume — target them more.")

# ══════════════════════════════════════════════════════════════
# TAB 10 — CATEGORY DEEP-DIVE ★
# ══════════════════════════════════════════════════════════════
with tabs[9]:
    sec("🔎 Tab 10 — Category Deep-Dive ★")
    info("Cross-tabulation: which combinations of features produce highest subscribe rates?")

    st.markdown("---")
    sec("📊 Job × Contact Method — Subscribe Rate Heatmap")
    if "contact" in df.columns and "job" in df.columns:
        heat = df.groupby(["job","contact"])[TARGET].mean().unstack() * 100
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(heat.round(1), annot=True, fmt=".1f", cmap="RdYlGn",
                    ax=ax, linewidths=0.5, annot_kws={"size": 9})
        ax.set_title("Subscribe Rate % — Job × Contact Method", fontsize=12, fontweight="bold")
        plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        sec("📊 Education × Marital — Subscribe Rate")
        if "education" in df.columns and "marital" in df.columns:
            heat2 = df.groupby(["education","marital"])[TARGET].mean().unstack() * 100
            fig2, ax2 = plt.subplots(figsize=(7, 4))
            sns.heatmap(heat2.round(1), annot=True, fmt=".1f", cmap="RdYlGn",
                        ax=ax2, linewidths=0.5, annot_kws={"size": 9})
            ax2.set_title("Subscribe Rate % — Education × Marital", fontsize=10, fontweight="bold")
            plt.tight_layout(); st.pyplot(fig2); plt.close()

    with col2:
        sec("📊 Season × Contact Method")
        if "season" in df.columns and "contact" in df.columns:
            heat3 = df.groupby(["season","contact"])[TARGET].mean().unstack() * 100
            fig3, ax3 = plt.subplots(figsize=(7, 4))
            sns.heatmap(heat3.round(1), annot=True, fmt=".1f", cmap="RdYlGn",
                        ax=ax3, linewidths=0.5, annot_kws={"size": 9})
            ax3.set_title("Subscribe Rate % — Season × Contact Method", fontsize=10, fontweight="bold")
            plt.tight_layout(); st.pyplot(fig3); plt.close()

    insight("Retired + cellular contact = highest subscribe rate combination.")
    insight("Autumn + cellular = best seasonal contact strategy.")
    warn("Telephone contact underperforms cellular in every job category — phase out telephone outreach.")

# ══════════════════════════════════════════════════════════════
# TAB 11 — STATISTICAL TESTS ★
# ══════════════════════════════════════════════════════════════
with tabs[10]:
    sec("🧪 Tab 11 — Statistical Tests ★")
    info("4 hypothesis tests validating the most important business questions.")

    def run_test(name, gA, gB, labelA, labelB):
        t_stat, p_val = stats.ttest_ind(gA.dropna(), gB.dropna(), equal_var=False)
        pooled   = np.sqrt((gA.std()**2 + gB.std()**2) / 2)
        cohens_d = (gA.mean() - gB.mean()) / (pooled + 1e-10)
        result_df = pd.DataFrame({
            "Metric": ["Test","Group A","Group B","A Mean","B Mean",
                       "t-stat","p-value","Significant","Cohen's d","Effect Size","Decision"],
            "Result": [
                "Welch T-Test",
                f"{labelA} (n={len(gA):,})",
                f"{labelB} (n={len(gB):,})",
                f"{gA.mean():.4f}",
                f"{gB.mean():.4f}",
                f"{t_stat:.4f}",
                f"{p_val:.6f}",
                "✅ YES" if p_val < 0.05 else "❌ NO",
                f"{cohens_d:.4f}",
                "Large" if abs(cohens_d)>0.8 else "Medium" if abs(cohens_d)>0.5 else "Small",
                "✅ REJECT H₀" if p_val < 0.05 else "❌ FAIL to reject H₀"
            ]
        })
        return result_df, p_val, cohens_d

    # T1: Cellular vs Telephone — balance
    st.markdown("---")
    sec("T1 — Cellular vs Telephone: Does contact method affect client balance?")
    if "contact" in df.columns:
        gA = df[df["contact"]=="cellular"]["balance"]
        gB = df[df["contact"]=="telephone"]["balance"]
        r1, p1, d1 = run_test("T1", gA, gB, "Cellular", "Telephone")
        col1, col2 = st.columns([1.2, 1])
        with col1:
            st.dataframe(r1, use_container_width=True)
        with col2:
            fig, ax = plt.subplots(figsize=(6,3))
            ax.hist(gA.clip(-500,5000), bins=40, alpha=0.6, color=CLR["primary"],
                    density=True, label="Cellular")
            ax.hist(gB.clip(-500,5000), bins=40, alpha=0.6, color=CLR["warning"],
                    density=True, label="Telephone")
            ax.set_title("T1: Balance by Contact Method"); ax.legend()
            plt.tight_layout(); st.pyplot(fig); plt.close()
        if p1 < 0.05:
            insight(f"T1: Cellular clients have significantly different balance (d={d1:.3f}) → richer clients prefer mobile.")

    # T2: Subscribed vs Not — balance
    st.markdown("---")
    sec("T2 — Subscribed vs Not Subscribed: Is balance significantly different?")
    gA2 = df_sub["balance"]
    gB2 = df_nos["balance"]
    r2, p2, d2 = run_test("T2", gA2, gB2, "Subscribed", "Not Subscribed")
    col1, col2 = st.columns([1.2, 1])
    with col1:
        st.dataframe(r2, use_container_width=True)
    with col2:
        fig2, ax2 = plt.subplots(figsize=(6,3))
        ax2.hist(gB2.clip(-500,5000), bins=40, alpha=0.6, color=CLR["grey"],
                 density=True, label="Not Subscribed")
        ax2.hist(gA2.clip(-500,5000), bins=40, alpha=0.7, color=CLR["success"],
                 density=True, label="Subscribed")
        ax2.set_title("T2: Balance by Subscription"); ax2.legend()
        plt.tight_layout(); st.pyplot(fig2); plt.close()
    if p2 < 0.05:
        insight(f"T2: Subscribed clients have significantly higher balance (d={d2:.3f}).")

    # T3: High vs Low campaign effort — balance
    st.markdown("---")
    sec("T3 — High Campaign Effort (>3 calls) vs Efficient (≤3 calls): Age difference?")
    if "high_campaign_effort" in df.columns:
        gA3 = df[df["high_campaign_effort"]==1]["age"]
        gB3 = df[df["high_campaign_effort"]==0]["age"]
        r3, p3, d3 = run_test("T3", gA3, gB3, ">3 Calls", "≤3 Calls")
        st.dataframe(r3, use_container_width=True)
        if p3 < 0.05:
            insight(f"T3: Age differs between high-effort and efficient contacts (d={d3:.3f}).")

    # T4: Previous success vs no previous — campaign count
    st.markdown("---")
    sec("T4 — Previous Success vs No Previous Contact: Campaign calls needed?")
    if "prev_success" in df.columns:
        gA4 = df[df["prev_success"]==1]["campaign"]
        gB4 = df[df["prev_success"]==0]["campaign"]
        r4, p4, d4 = run_test("T4", gA4, gB4, "Prev Success", "No Prev Success")
        st.dataframe(r4, use_container_width=True)
        if p4 < 0.05:
            insight(f"T4: Clients with previous success need fewer calls (d={d4:.3f}) — warm leads convert faster.")

# ══════════════════════════════════════════════════════════════
# TAB 12 — FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════
with tabs[11]:
    sec("⚙️ Tab 12 — Feature Engineering")

    fe = pd.DataFrame({
        "Feature":   ["age_group","balance_category","was_contacted_before",
                      "high_campaign_effort","prev_success","has_debt",
                      "season","month_num","*_enc columns"],
        "Source":    ["age cut into 6 bands",
                      "balance cut into 5 bands",
                      "pdays != -1",
                      "campaign > 3",
                      "poutcome == 'success'",
                      "housing='yes' OR loan='yes'",
                      "month binned to season",
                      "month mapped to 1-12",
                      "LabelEncoder on all categoricals"],
        "Reason":    ["Non-linear age effect on subscription",
                      "Wealth segmentation — richer subscribe more",
                      "Previously warmed leads convert better",
                      "High effort signals resistant client",
                      "Strongest predictor — past success = future success",
                      "Debt burden reduces financial capacity",
                      "Seasonal pattern in subscription rates",
                      "Ordered month for correlation analysis",
                      "ML-ready categorical features"],
    })
    st.dataframe(fe, use_container_width=True)

    insight("prev_success is the single most powerful engineered feature — business logic validated by data.")
    warn("duration was dropped — it is only known AFTER the call, making it leakage for predictive models.")

# ══════════════════════════════════════════════════════════════
# TAB 13 — INSIGHTS & REPORT
# ══════════════════════════════════════════════════════════════
with tabs[12]:
    sec("💡 Tab 13 — Insights & Recommendations")

    st.markdown(f"### 🏦 Bank Marketing — Final Report")
    st.markdown(f"**{len(df):,} clients · {sub_rate:.1f}% subscribe rate · UCI Bank · M3**")
    st.markdown("---")

    sec("1️⃣ Top Subscription Drivers")
    insight("Previous campaign success — by far the strongest predictor (3-4× base rate).")
    insight("Cellular contact — outperforms telephone in every segment.")
    insight("High account balance — financial capacity enables investment decisions.")
    insight("March, September, October, December — seasonally high conversion months.")
    insight("Students and retired — highest subscribe rates (time + financial planning).")

    st.markdown("---")
    sec("2️⃣ Recommendations")
    recs = [
        ("🎯 Target Warm Leads First",
         "Clients who subscribed in previous campaigns should be the primary target — 3-4× conversion."),
        ("📱 Switch to Cellular Only",
         "Telephone contact underperforms in every segment — reallocate budget to cellular outreach."),
        ("💰 Balance-Based Targeting",
         "Focus on clients with balance > €1,000 — higher capacity = higher subscription rate."),
        ("📅 Optimal Timing",
         "Launch campaigns in March, September, October — avoid May (high volume, lowest conversion)."),
        ("⏱ Stop at 3 Calls",
         "Subscription rate drops sharply after 3 calls — stop pursuing resistant clients, redirect resources."),
        ("👴 Segment Retirees",
         "Create dedicated retirement investment product campaign — highest ROI demographic."),
    ]
    for title, text in recs:
        st.markdown(f'<div class="warn-box"><p><b>{title}:</b> {text}</p></div>',
                    unsafe_allow_html=True)

    st.markdown("---")
    report_txt = f"""BANK MARKETING — FINAL REPORT
M3 · UCI Bank Marketing · {len(df):,} Clients
Subscribe Rate: {sub_rate:.1f}% | Severe Imbalance → class_weight='balanced'

TOP SUBSCRIPTION DRIVERS:
1. Previous campaign success (poutcome=success) — strongest predictor
2. Cellular contact method — outperforms telephone
3. High account balance — financial capacity
4. Seasonal: March, September, October, December
5. Student and retired segments — highest rates

KEY RECOMMENDATIONS:
- Target previous subscribers first (warm leads)
- Switch all outreach to cellular contact only
- Focus on balance > €1,000 clients
- Campaign timing: Mar, Sep, Oct, Dec
- Stop at 3 calls — efficiency drops sharply after
- Dedicated campaign for retired clients (highest ROI)

TECHNICAL NOTE:
- 'duration' column DROPPED (post-call leakage)
- class_weight='balanced' mandatory (11.7% subscribe rate)
- Evaluate with F1, Recall, ROC-AUC only
"""
    col1, col2 = st.columns(2)
    with col1:
        st.download_button("📥 Download Report (.txt)", report_txt,
                           file_name="BankMarketing_Report_M3.txt",
                           mime="text/plain", use_container_width=True)
    with col2:
        st.download_button("📥 Download Clean Data (.csv)",
                           df.to_csv(index=False),
                           file_name="bank_clean_M3.csv",
                           mime="text/csv", use_container_width=True)
