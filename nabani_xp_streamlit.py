# -*- coding: utf-8 -*-
"""
Created on Thu Jan 29 22:12:34 2026

@author: naban
"""

# ==================================================
# 0. IMPORT LIBRARIES
# ==================================================
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import statsmodels.api as sm
import numpy as np

# --------------------------------------------------
# Global matplotlib settings (journal-style)
# --------------------------------------------------
plt.rcParams.update({
    "figure.figsize": (5, 3),
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9
})

# --------------------------------------------------
# Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Research Profile and Result Outputs",
    layout="wide"
)

# ==================================================
# HEADER SECTION
# ==================================================
st.title("Research data analysis")

st.markdown("### By XP Nabani")
st.markdown("**Research Field:** A postgraduate student in the field of Aquatic Ecology and Ecophysiology")

st.markdown("""
**About researcher and research interests**  
I am a researcher in aquatic ecology and ecophysiology, with interests in how
environmental variability, particularly temperature affects physiological
performance, metabolic capacity, and ecological resilience of aquatic organisms.
Additionally, I have recently become interested in intergrating data science skills
with ecological/biological skills to answer complex ecological/biologicl questions.
""")

st.markdown("""
**Introduction**  
This project presents a subsetdata from my previous research and demonstrates how I
apporach data analyses the dataset, from exploratory data analysis (EDA) through to 
statistical modelling (this is a simple but not the entire approach).
""")

st.markdown("---")

# ==================================================
# 1. DATA LOADING
# ==================================================
st.header("1. Libraries and Data Loading")

st.markdown("""
**The following libraries were used for data manipulation, visualization, and
statistical modelling.**
""")

st.code("""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import statsmodels.api as sm
import numpy as np
""", language="python")

# Load data
df = pd.read_csv(
    "met_data.csv",
    sep=";",
    encoding="utf-8-sig"
)

# Clean column names (VERY important)
df.columns = df.columns.str.strip()

st.markdown("**First rows of the dataset:**")
st.dataframe(df.head())

st.markdown("**Dataset information:**")
st.text(df.info())

st.markdown(f"**Dataset dimensions:** {df.shape}")

st.markdown("---")

# ==================================================
# 2. DATA CLEANING AND TRANSFORMATION
# ==================================================
st.header("2. Data Cleaning and Transformation")

st.markdown("""
**Rows with missing maximum metabolic rate (MMR) values were removed to ensure valid
statistical inference. Variables were converted to appropriate data types prior to
analysis.**
""")

# Remove missing MMR
df = df.dropna(subset=["MMR"])
st.markdown(f"**Dataset dimensions after cleaning:** {df.shape}")

# Convert variable types
df["Temperature"] = df["Temperature"].astype(float)
df["Site"] = df["Site"].astype("category")
df["FishID"] = df["FishID"].astype("category")

st.markdown("---")

# ==================================================
# 3. EXPLORATORY DATA ANALYSIS (EDA)
# ==================================================
st.header("3. Exploratory Data Analysis (EDA)")

# ---- Mean MMR by Site ----
st.subheader("Mean MMR by Site")

st.markdown("""
**This plot summarises differences in mean maximum metabolic rate (MMR) between sampling
sites, with error bars representing standard deviation.**
""")

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    fig1, ax1 = plt.subplots(figsize=(5, 3))
    sns.barplot(data=df, x="Site", y="MMR", errorbar="sd", ax=ax1)
    ax1.set_title("Mean MMR by Site")
    st.pyplot(fig1)

# ---- MMR vs Temperature ----
st.subheader("MMR vs Temperature ")

st.markdown("""
**This scatter plot shows the relationship between temperature and MMR, with points
coloured by site to highlight potential site-specific patterns.**
""")

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    fig2, ax2 = plt.subplots(figsize=(5, 3))
    sns.scatterplot(data=df, x="Temperature", y="MMR", hue="Site", ax=ax2)
    ax2.set_title("MMR vs Temperature ")
    ax2.set_xlabel("Temperature (°C)")
    ax2.set_ylabel(r'MMR($O_2$) ~ mg $\cdot$ min$^{-1}$ $\cdot$ kg$^{-1}$')
    st.pyplot(fig2)

# ---- Individual-level variation ----
st.subheader("Individual-level MMR Variation")

st.markdown("""
**Individual-level variation in MMR within and among sites is shown using a strip plot,
highlighting within-site variability among fish.**
""")

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    fig3, ax3 = plt.subplots(figsize=(5, 3))
    sns.stripplot(data=df, x="Site", y="MMR", jitter=True, ax=ax3)
    ax3.set_title("Fish-level MMR variation by Site")
    st.pyplot(fig3)

st.markdown("---")

# ==================================================
# 4. STATISTICAL ANALYSES AND MODELLING
# ==================================================
st.header("4. Statistical Analyses and Modelling")

st.markdown("""
 **A linear mixed-effects model was fitted to test the effect of temperature and
its quadratic term on MMR, while accounting for repeated measures at the individual 
level (FishID) and variation among sites.**
""")

st.code("""
model = smf.mixedlm(
    "MMR ~ Temperature + Temp2 + Site + Temperature:Site + Temp2:Site",
    data=df,
    groups="FishID",
    re_formula="~Temperature"
)

result = model.fit()
result.summary()
""", language="python")
# Add a polynomial/quadratic term
df["Temp2"] = df["Temperature"]**2

# Fit model
model = smf.mixedlm(
    "MMR ~ Temperature + Temp2 + Site + Temperature:Site + Temp2:Site",
    data=df,
    groups="FishID",
    re_formula="~Temperature"
)

result_model = model.fit()
st.subheader("Model summary")
st.code(result_model.summary().as_text())


st.markdown("---")

# ==================================================
# MODEL DIAGNOSTICS (PLOTTED TOGETHER)
# ==================================================
st.subheader("5. Model Diagnostics")

st.markdown(""" 
**Model diagnostics were used to assess homoscedasticity, normality of residuals,
and overall model fit.**
""")

fitted = result_model.fittedvalues
residuals = result_model.resid

col1, col2, col3 = st.columns(3)

# Residuals vs fitted
with col1:
    fig4, ax4 = plt.subplots(figsize=(4, 4))
    ax4.scatter(fitted, residuals)
    ax4.axhline(0, linestyle="--", color="red")
    ax4.set_xlabel("Fitted values")
    ax4.set_ylabel("Residuals")
    ax4.set_title("Residuals vs Fitted")
    st.pyplot(fig4)

# Q-Q plot
with col2:
    fig5 = plt.figure(figsize=(4, 4))
    sm.qqplot(residuals, line="45", ax=plt.gca())
    plt.title("Q–Q Plot")
    st.pyplot(fig5)

# Residual distribution
with col3:
    fig6, ax6 = plt.subplots(figsize=(4, 4))
    sns.histplot(residuals, kde=True, ax=ax6)
    ax6.set_title("Residuals")
    st.pyplot(fig6)

st.markdown("---")

# ==================================================
# MODEL PERFORMANCE
# ==================================================
st.subheader("6. Model Performance")

var_fixed = np.var(result_model.fittedvalues)
var_resid = result_model.scale

if result_model.cov_re.shape[0] > 0:
    var_random = result_model.cov_re.values[0, 0]
else:
    var_random = 0.0

r2_marginal = var_fixed / (var_fixed + var_random + var_resid)
r2_conditional = (var_fixed + var_random) / (var_fixed + var_random + var_resid)

st.markdown(""" 
**Marginal R² represents variance explained by fixed effects only, whereas conditional R²
represents variance explained by both fixed and random effects.**
""")

st.write(f"**Marginal R²:** {r2_marginal:.3f}")
st.write(f"**Conditional R²:** {r2_conditional:.3f}")

st.markdown("---")


# ==================================================
# Predictions
# ==================================================
st.header("7. Predictions and Model Visualization")

temp_range = np.linspace(df["Temperature"].min(), df["Temperature"].max(), 100)

pred_df = pd.DataFrame({
    "Temperature": np.tile(temp_range, df["Site"].nunique()),
    "Site": np.repeat(df["Site"].unique(), 100)
})

pred_df["Temp2"] = pred_df["Temperature"]**2
pred_df["MMR_pred"] = result_model.predict(pred_df)

st.markdown(""" 
**A visual presentation of model predictions. The figure shows that temperature
effct on MMR is not always linear**
""")
fig, ax = plt.subplots(figsize=(5, 3))

# raw points
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    fig, ax = plt.subplots(figsize=(5, 3))
    sns.scatterplot(
        data=df,
        x="Temperature",
        y="MMR",
        hue="Site",
        alpha=0.5,
        ax=ax
    )
    sns.lineplot(
        data=pred_df,
        x="Temperature",
        y="MMR_pred",
        hue="Site",
        linewidth=3,
        legend=False,
        ax=ax
    )
    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel(r'MMR($O_2$) ~ mg $\cdot$ min$^{-1}$ $\cdot$ kg$^{-1}$')
    ax.set_title("Raw data and mixed model predictions")
    st.pyplot(fig)
st.success("Analysis complete.")


# ==================================================
# FOOTER
# ==================================================
st.markdown("---")
st.write("© 2026 XP Nabani | Email: nabanixp@gmail.com | LinkedIn: https://www.linkedin.com/in/xolani-prince-nabani-9a33b2112/")


