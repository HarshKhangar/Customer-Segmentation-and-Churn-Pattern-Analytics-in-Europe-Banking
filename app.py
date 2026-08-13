import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="European Bank Churn Analytics",
    page_icon="🏦",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("🏦 European Bank Customer Churn Analytics")

st.markdown("""
### Customer Segmentation & Churn Pattern Analytics

This application analyzes European banking customers and identifies
high-risk customer segments using demographic, financial and
engagement-related factors.
""")


# ============================================================
# DATA LOADING
# ============================================================

@st.cache_data
def load_data(uploaded_file):

    df = pd.read_csv(uploaded_file)

    return df


# ============================================================
# DATA VALIDATION
# ============================================================

def validate_data(df):

    expected_cols = {
        "CustomerId",
        "Surname",
        "CreditScore",
        "Geography",
        "Gender",
        "Age",
        "Tenure",
        "Balance",
        "NumOfProducts",
        "HasCrCard",
        "IsActiveMember",
        "EstimatedSalary",
        "Exited"
    }

    missing = expected_cols - set(df.columns)

    if missing:
        return False, f"Missing columns: {missing}"

    return True, "All expected columns are present."


# ============================================================
# DATA CLEANING
# ============================================================

def clean_data(df):

    df = df.copy()

    # Remove non-analytical columns
    drop_cols = [
        col for col in ["Surname", "RowNumber"]
        if col in df.columns
    ]

    df.drop(columns=drop_cols, inplace=True, errors="ignore")

    # Data types
    df["Geography"] = df["Geography"].astype("category")
    df["Gender"] = df["Gender"].astype("category")

    df["HasCrCard"] = df["HasCrCard"].astype(int)
    df["IsActiveMember"] = df["IsActiveMember"].astype(int)
    df["Exited"] = df["Exited"].astype(int)

    # Remove exact duplicates
    df.drop_duplicates(inplace=True)

    return df


# ============================================================
# CUSTOMER SEGMENTATION
# ============================================================

def create_segments(df):

    df = df.copy()

    # Age Group
    df["AgeGroup"] = pd.cut(
        df["Age"],
        bins=[0, 30, 45, 60, np.inf],
        labels=[
            "<30",
            "30-45",
            "46-60",
            "60+"
        ]
    )

    # Credit Score Band
    df["CreditScoreBand"] = pd.cut(
        df["CreditScore"],
        bins=[0, 580, 700, np.inf],
        labels=[
            "Low (<580)",
            "Medium (580-700)",
            "High (>700)"
        ]
    )

    # Tenure Group
    df["TenureGroup"] = pd.cut(
        df["Tenure"],
        bins=[-1, 2, 6, np.inf],
        labels=[
            "New (0-2 yrs)",
            "Mid-term (3-6 yrs)",
            "Long-term (7+ yrs)"
        ]
    )

    # Balance Segment
    def balance_bucket(balance):

        if balance == 0:
            return "Zero-balance"

        elif balance < 100000:
            return "Low-balance"

        else:
            return "High-balance"

    df["BalanceSegment"] = df["Balance"].apply(
        balance_bucket
    )

    return df


# ============================================================
# KPI CALCULATION
# ============================================================

def calculate_kpis(df):

    total_customers = len(df)

    churned_customers = df["Exited"].sum()

    retained_customers = total_customers - churned_customers

    churn_rate = (
        df["Exited"].mean() * 100
    )

    # High-value threshold
    balance_threshold = df["Balance"].quantile(0.75)

    high_value = df[
        df["Balance"] >= balance_threshold
    ]

    high_value_churn_rate = (
        high_value["Exited"].mean() * 100
        if len(high_value) > 0
        else 0
    )

    high_value_churned = high_value[
        high_value["Exited"] == 1
    ]

    balance_at_risk = (
        high_value_churned["Balance"].sum()
    )

    return {
        "total_customers": total_customers,
        "churned_customers": churned_customers,
        "retained_customers": retained_customers,
        "churn_rate": churn_rate,
        "balance_threshold": balance_threshold,
        "high_value_customers": len(high_value),
        "high_value_churn_rate": high_value_churn_rate,
        "balance_at_risk": balance_at_risk
    }


# ============================================================
# SEGMENT ANALYSIS
# ============================================================

def segment_analysis(df, column):

    result = (
        df.groupby(
            column,
            observed=True
        )
        .agg(
            CustomerCount=("Exited", "size"),
            ChurnedCount=("Exited", "sum")
        )
        .reset_index()
    )

    result["ChurnRate(%)"] = (
        result["ChurnedCount"]
        / result["CustomerCount"]
        * 100
    ).round(2)

    total_churners = df["Exited"].sum()

    result["ShareOfChurners(%)"] = (
        result["ChurnedCount"]
        / total_churners
        * 100
    ).round(2)

    result = result.sort_values(
        "ChurnRate(%)",
        ascending=False
    )

    return result


# ============================================================
# GEOGRAPHY × AGE
# ============================================================

def geography_age_analysis(df):

    table = pd.pivot_table(
        df,
        values="Exited",
        index="Geography",
        columns="AgeGroup",
        aggfunc="mean",
        observed=True
    ) * 100

    return table.round(2)


# ============================================================
# MAIN APPLICATION
# ============================================================

st.sidebar.header("📂 Dataset")

uploaded_file = st.sidebar.file_uploader(
    "Upload European Bank CSV",
    type=["csv"]
)


# ============================================================
# LOAD DATA
# ============================================================

if uploaded_file is not None:

    df = load_data(uploaded_file)

else:

    try:
        df = pd.read_csv("european_bank.csv")

    except FileNotFoundError:

        st.warning(
            "Please upload the European Bank CSV dataset."
        )

        st.stop()


# ============================================================
# VALIDATION
# ============================================================

valid, message = validate_data(df)

if not valid:

    st.error(message)

    st.stop()

else:

    st.success(message)


# ============================================================
# CLEAN + SEGMENT
# ============================================================

df = clean_data(df)

df = create_segments(df)


# ============================================================
# KPI CALCULATION
# ============================================================

kpis = calculate_kpis(df)


# ============================================================
# SIDEBAR FILTERS
# ============================================================

st.sidebar.header("🔎 Filters")


geographies = st.sidebar.multiselect(
    "Geography",
    options=df["Geography"].dropna().unique(),
    default=list(
        df["Geography"].dropna().unique()
    )
)


genders = st.sidebar.multiselect(
    "Gender",
    options=df["Gender"].dropna().unique(),
    default=list(
        df["Gender"].dropna().unique()
    )
)


age_groups = st.sidebar.multiselect(
    "Age Group",
    options=df["AgeGroup"].dropna().unique(),
    default=list(
        df["AgeGroup"].dropna().unique()
    )
)


# ============================================================
# APPLY FILTERS
# ============================================================

filtered_df = df[
    df["Geography"].isin(geographies)
    &
    df["Gender"].isin(genders)
    &
    df["AgeGroup"].isin(age_groups)
]


# ============================================================
# TABS
# ============================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "📊 Overview",
        "👥 Segmentation",
        "📈 Churn Analysis",
        "💰 High-Value Customers",
        "📥 Data & Downloads"
    ]
)


# ============================================================
# TAB 1 — OVERVIEW
# ============================================================

with tab1:

    st.header("📊 Banking Churn Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Customers",
        f"{kpis['total_customers']:,}"
    )

    col2.metric(
        "Churned Customers",
        f"{kpis['churned_customers']:,}"
    )

    col3.metric(
        "Overall Churn Rate",
        f"{kpis['churn_rate']:.2f}%"
    )

    col4.metric(
        "Retained Customers",
        f"{kpis['retained_customers']:,}"
    )


    st.divider()

    # Churn Pie Chart

    churn_counts = df["Exited"].value_counts()

    fig, ax = plt.subplots()

    ax.pie(
        churn_counts.values,
        labels=["Retained", "Churned"],
        autopct="%1.1f%%"
    )

    ax.set_title("Overall Customer Churn")

    st.pyplot(fig)


# ============================================================
# TAB 2 — SEGMENTATION
# ============================================================

with tab2:

    st.header("👥 Customer Segmentation")

    st.subheader("Customer Dataset")

    st.dataframe(
        filtered_df,
        use_container_width="stretch"
    )

    st.subheader("Age Group Distribution")

    age_result = segment_analysis(
        filtered_df,
        "AgeGroup"
    )

    st.dataframe(
        age_result,
        use_container_width="stretch"
    )


# ============================================================
# TAB 3 — CHURN ANALYSIS
# ============================================================

with tab3:

    st.header("📈 Churn Analysis")

    # Geography

    st.subheader("Churn Rate by Geography")

    geo_result = segment_analysis(
        filtered_df,
        "Geography"
    )

    fig, ax = plt.subplots()

    sns.barplot(
        data=geo_result,
        x="Geography",
        y="ChurnRate(%)",
        ax=ax
    )

    ax.set_ylabel("Churn Rate (%)")

    st.pyplot(fig)


    # Age

    st.subheader("Churn Rate by Age Group")

    age_result = segment_analysis(
        filtered_df,
        "AgeGroup"
    )

    fig, ax = plt.subplots()

    sns.barplot(
        data=age_result,
        x="AgeGroup",
        y="ChurnRate(%)",
        ax=ax
    )

    ax.set_ylabel("Churn Rate (%)")

    st.pyplot(fig)


    # Credit Score

    st.subheader("Churn Rate by Credit Score")

    credit_result = segment_analysis(
        filtered_df,
        "CreditScoreBand"
    )

    fig, ax = plt.subplots()

    sns.barplot(
        data=credit_result,
        x="CreditScoreBand",
        y="ChurnRate(%)",
        ax=ax
    )

    ax.set_ylabel("Churn Rate (%)")

    st.pyplot(fig)


    # Tenure

    st.subheader("Churn Rate by Tenure")

    tenure_result = segment_analysis(
        filtered_df,
        "TenureGroup"
    )

    fig, ax = plt.subplots()

    sns.barplot(
        data=tenure_result,
        x="TenureGroup",
        y="ChurnRate(%)",
        ax=ax
    )

    ax.set_ylabel("Churn Rate (%)")

    st.pyplot(fig)


    # Balance

    st.subheader("Churn Rate by Balance Segment")

    balance_result = segment_analysis(
        filtered_df,
        "BalanceSegment"
    )

    fig, ax = plt.subplots()

    sns.barplot(
        data=balance_result,
        x="BalanceSegment",
        y="ChurnRate(%)",
        ax=ax
    )

    ax.set_ylabel("Churn Rate (%)")

    st.pyplot(fig)


    # Geography × Age Heatmap

    st.subheader("🌡️ Geography × Age Churn Heatmap")

    geo_age = geography_age_analysis(
        filtered_df
    )

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    sns.heatmap(
        geo_age,
        annot=True,
        fmt=".2f",
        ax=ax
    )

    ax.set_xlabel("Age Group")

    ax.set_ylabel("Geography")

    st.pyplot(fig)


# ============================================================
# TAB 4 — HIGH-VALUE CUSTOMERS
# ============================================================

with tab4:

    st.header("💰 High-Value Customer Analysis")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "High-Value Customers",
        f"{kpis['high_value_customers']:,}"
    )

    col2.metric(
        "High-Value Churn",
        f"{kpis['high_value_churn_rate']:.2f}%"
    )

    col3.metric(
        "Balance at Risk",
        f"₹{kpis['balance_at_risk']:,.2f}"
    )


    st.info(
        "High-value customers are defined as customers "
        "whose balance is at or above the 75th percentile."
    )


    high_value_df = df[
        df["Balance"] >= kpis["balance_threshold"]
    ]

    st.subheader("High-Value Customer Records")

    st.dataframe(
        high_value_df,
        use_container_width="stretch"
    )


# ============================================================
# TAB 5 — DATA & DOWNLOADS
# ============================================================

with tab5:

    st.header("📥 Data & Downloads")

    st.subheader("Processed Dataset")

    st.dataframe(
        df,
        width="stretch"
    )


    # Processed dataset

    csv_data = df.to_csv(
        index=False
    ).encode("utf-8")


    st.download_button(
        label="⬇️ Download Segmented Customer Data",
        data=csv_data,
        file_name="segmented_customer_data.csv",
        mime="text/csv"
    )


    # KPI data

    kpi_df = pd.DataFrame(
        {
            "KPI": [
                "Total Customers",
                "Churned Customers",
                "Retained Customers",
                "Overall Churn Rate (%)",
                "High-Value Customers",
                "High-Value Churn Rate (%)",
                "Balance at Risk"
            ],

            "Value": [
                kpis["total_customers"],
                kpis["churned_customers"],
                kpis["retained_customers"],
                round(kpis["churn_rate"], 2),
                kpis["high_value_customers"],
                round(
                    kpis["high_value_churn_rate"],
                    2
                ),
                round(
                    kpis["balance_at_risk"],
                    2
                )
            ]
        }
    )


    kpi_csv = kpi_df.to_csv(
        index=False
    ).encode("utf-8")


    st.download_button(
        label="⬇️ Download KPI Summary",
        data=kpi_csv,
        file_name="kpi_summary.csv",
        mime="text/csv"
    )