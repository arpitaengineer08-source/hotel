

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from data_cleaning import load_raw_data, clean_data, RAW_PATH, CLEAN_PATH

sns.set_style("whitegrid")

st.set_page_config(page_title="Hotel Business Dashboard", layout="wide", page_icon="🏨")


# ----------------------------------------------------------------------
# Data loading (cached so cleaning only runs once per session)
# ----------------------------------------------------------------------
@st.cache_data
def get_data() -> pd.DataFrame:
    if os.path.exists(CLEAN_PATH):
        df = pd.read_csv(CLEAN_PATH)
    else:
        df_raw = load_raw_data(RAW_PATH)
        df = clean_data(df_raw)
        os.makedirs("data", exist_ok=True)
        df.to_csv(CLEAN_PATH, index=False)

    month_order = [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December",
    ]
    df["arrival_date_month"] = pd.Categorical(
        df["arrival_date_month"], categories=month_order, ordered=True
    )
    stay_order = ["1 night", "2-3 nights", "4-7 nights", "8-14 nights", "15+ nights"]
    df["stay_duration_bucket"] = pd.Categorical(
        df["stay_duration_bucket"], categories=stay_order, ordered=True
    )
    lead_order = ["0-7 days", "8-30 days", "31-90 days", "91-180 days", "181-365 days", "365+ days"]
    df["lead_time_bucket"] = pd.Categorical(
        df["lead_time_bucket"], categories=lead_order, ordered=True
    )
    return df


df = get_data()

# ----------------------------------------------------------------------
# Sidebar filters
# ----------------------------------------------------------------------
st.sidebar.header("Filters")

hotel_options = ["All"] + sorted(df["hotel"].unique().tolist())
hotel_filter = st.sidebar.selectbox("Hotel type", hotel_options)

year_options = sorted(df["arrival_date_year"].unique().tolist())
year_filter = st.sidebar.multiselect("Arrival year", year_options, default=year_options)

status_options = ["All", "Canceled", "Not Canceled"]
status_filter = st.sidebar.selectbox("Booking status", status_options)

st.sidebar.markdown("---")
st.sidebar.caption(f"Dataset: {len(df):,} cleaned bookings, {df['arrival_date_year'].min()}–{df['arrival_date_year'].max()}")

filtered = df.copy()
if hotel_filter != "All":
    filtered = filtered[filtered["hotel"] == hotel_filter]
if year_filter:
    filtered = filtered[filtered["arrival_date_year"].isin(year_filter)]
if status_filter != "All":
    filtered = filtered[filtered["is_canceled_label"] == status_filter]

# ----------------------------------------------------------------------
# Header + KPIs
# ----------------------------------------------------------------------
st.title("🏨 Investigate Hotel Business — Booking & Cancellation Dashboard")
st.caption("Data Visualization Project · Hotel bookings 2017–2019 · Cleaned dataset")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Bookings", f"{len(filtered):,}")
k2.metric("Cancellation Rate", f"{filtered['is_canceled'].mean()*100:.1f}%")
k3.metric("Avg. Lead Time", f"{filtered['lead_time'].mean():.0f} days")
k4.metric("Avg. Daily Rate (ADR)", f"${filtered['adr'].mean():.2f}")

st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs([
    "1️⃣ Hotel Type & Seasonality",
    "2️⃣ Stay Duration vs Cancellation",
    "3️⃣ Lead Time vs Cancellation",
    "📋 Cleaned Data & Summary",
])

# ----------------------------------------------------------------------
# TAB 1: Hotel type share + monthly seasonality
# ----------------------------------------------------------------------
with tab1:
    st.subheader("Which hotel type do customers book most often?")
    c1, c2 = st.columns([1, 2])

    with c1:
        counts = filtered["hotel"].value_counts()
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.pie(counts, labels=counts.index, autopct="%1.1f%%",
               colors=["#4C72B0", "#DD8452"], startangle=90)
        ax.set_title("Share of Bookings by Hotel Type")
        st.pyplot(fig)
        st.caption("City Hotel typically accounts for the larger share of total bookings.")

    with c2:
        monthly = (
            filtered.groupby(["arrival_date_month", "hotel"], observed=True)
            .size().reset_index(name="bookings")
        )
        fig, ax = plt.subplots(figsize=(9, 5))
        sns.lineplot(data=monthly, x="arrival_date_month", y="bookings", hue="hotel", marker="o", ax=ax)
        ax.set_title("Monthly Bookings by Hotel Type")
        ax.set_xlabel("Month")
        ax.set_ylabel("Number of Bookings")
        ax.tick_params(axis="x", rotation=45)
        st.pyplot(fig)
        st.caption("Bookings for both hotel types peak in the summer months, dipping in winter.")

# ----------------------------------------------------------------------
# TAB 2: Stay duration vs cancellation
# ----------------------------------------------------------------------
with tab2:
    st.subheader("Does length of stay affect cancellation rate?")
    c1, c2 = st.columns(2)

    with c1:
        cancel_by_hotel = filtered.groupby("hotel")["is_canceled"].mean().mul(100)
        fig, ax = plt.subplots(figsize=(5, 5))
        sns.barplot(x=cancel_by_hotel.index, y=cancel_by_hotel.values,
                    hue=cancel_by_hotel.index, palette=["#4C72B0", "#DD8452"], legend=False, ax=ax)
        ax.set_title("Overall Cancellation Rate by Hotel Type")
        ax.set_ylabel("Cancellation Rate (%)")
        for i, v in enumerate(cancel_by_hotel.values):
            ax.text(i, v + 0.5, f"{v:.1f}%", ha="center")
        st.pyplot(fig)

    with c2:
        stay_cancel = (
            filtered.groupby(["stay_duration_bucket", "hotel"], observed=True)["is_canceled"]
            .mean().mul(100).reset_index(name="cancel_rate")
        )
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.barplot(data=stay_cancel, x="stay_duration_bucket", y="cancel_rate", hue="hotel", ax=ax)
        ax.set_title("Cancellation Rate by Length of Stay")
        ax.set_xlabel("Total Stay Duration")
        ax.set_ylabel("Cancellation Rate (%)")
        ax.tick_params(axis="x", rotation=20)
        st.pyplot(fig)

    st.caption("Cancellation rate tends to rise as the length of stay increases, more sharply for City Hotel bookings.")

# ----------------------------------------------------------------------
# TAB 3: Lead time vs cancellation
# ----------------------------------------------------------------------
with tab3:
    st.subheader("Does lead time affect cancellation rate?")

    lead_cancel = (
        filtered.groupby(["lead_time_bucket", "hotel"], observed=True)["is_canceled"]
        .mean().mul(100).reset_index(name="cancel_rate")
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(data=lead_cancel, x="lead_time_bucket", y="cancel_rate", hue="hotel", marker="o", ax=ax)
    ax.set_title("Cancellation Rate by Lead Time")
    ax.set_xlabel("Lead Time Bucket")
    ax.set_ylabel("Cancellation Rate (%)")
    st.pyplot(fig)
    st.caption("Cancellation rate is lowest for bookings made close to arrival and climbs as lead time grows, "
               "rising most sharply for City Hotel.")

    st.markdown("#### Lead time distribution")
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.histplot(data=filtered, x="lead_time", hue="hotel", bins=40, kde=False, ax=ax, multiple="layer", alpha=0.6)
    ax.set_title("Distribution of Lead Time")
    ax.set_xlabel("Lead Time (days)")
    st.pyplot(fig)

# ----------------------------------------------------------------------
# TAB 4: Cleaned data preview + summary
# ----------------------------------------------------------------------
with tab4:
    st.subheader("Cleaned dataset preview")
    st.dataframe(filtered.head(200), use_container_width=True)

    st.subheader("Key business recommendations")
    st.markdown("""
- **Hotel type & seasonality:** City Hotel drives the bulk of bookings; run targeted off-season
  promotions for Resort Hotel and add staffing/inventory ahead of the summer peak for both properties.
- **Stay duration:** Since cancellations rise with longer stays, consider tiered deposit or
  cancellation-fee policies for stays beyond ~7 nights, especially for City Hotel.
- **Lead time:** Since far-ahead bookings cancel more often, add pre-arrival reminder emails,
  partial non-refundable deposits, or free-rescheduling options for bookings made 90+ days out.
    """)

    st.download_button(
        "Download cleaned dataset (CSV)",
        data=filtered.to_csv(index=False).encode("utf-8"),
        file_name="hotel_bookings_clean_filtered.csv",
        mime="text/csv",
    )
