"""
app.py
-------
Streamlit dashboard for the "Investigate Hotel Business using Data
Visualization" project. Answers the three business questions interactively:

  1. Which hotel type is booked most often (and seasonality)?
  2. Does length of stay affect cancellation rate?
  3. Does lead time affect cancellation rate?

Run with:
    streamlit run app.py
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import streamlit.components.v1 as components

from data_cleaning import load_raw_data, clean_data, RAW_PATH, CLEAN_PATH

sns.set_style("whitegrid")

st.set_page_config(page_title="Hotel Business Dashboard", layout="wide", page_icon="🏨")

# ----------------------------------------------------------------------
# Global CSS: animated gradient header, KPI card hover/fade-in, chart
# hover zoom, smooth tab transitions, custom spinner colour.
# ----------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');

html, body, [class*="css"]  { font-family: 'Poppins', sans-serif; }

/* Fade the whole page content in on load */
[data-testid="stAppViewContainer"] > .main {
    animation: pageFadeIn 0.6s ease both;
}
@keyframes pageFadeIn {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* Animated gradient header banner */
.dash-header {
    background: linear-gradient(-45deg, #4C72B0, #6a89cc, #DD8452, #f6b93b);
    background-size: 400% 400%;
    animation: gradientShift 12s ease infinite;
    padding: 26px 32px;
    border-radius: 16px;
    margin-bottom: 22px;
    box-shadow: 0 8px 24px rgba(0,0,0,0.15);
}
@keyframes gradientShift {
    0%   { background-position: 0% 50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
.dash-header h1 {
    color: #fff; margin: 0; font-size: 1.9rem; font-weight: 700;
    animation: fadeSlideDown 0.8s ease both;
}
.dash-header p {
    color: rgba(255,255,255,0.92); margin: 6px 0 0 0; font-size: 0.95rem;
    animation: fadeSlideDown 1s ease both;
}
@keyframes fadeSlideDown {
    from { opacity: 0; transform: translateY(-12px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* KPI cards */
.kpi-row { display: flex; gap: 16px; margin-bottom: 4px; flex-wrap: wrap; }
.kpi-card {
    flex: 1; min-width: 190px; background: #ffffff; border-radius: 14px;
    padding: 16px 20px; box-shadow: 0 2px 10px rgba(0,0,0,0.08);
    border-left: 5px solid #4C72B0;
    transition: transform 0.25s ease, box-shadow 0.25s ease;
    animation: cardFadeUp 0.6s ease both;
}
.kpi-card:hover { transform: translateY(-6px) scale(1.02); box-shadow: 0 10px 24px rgba(0,0,0,0.16); }
.kpi-card:nth-child(1) { animation-delay: 0.05s; border-left-color: #4C72B0; }
.kpi-card:nth-child(2) { animation-delay: 0.15s; border-left-color: #DD8452; }
.kpi-card:nth-child(3) { animation-delay: 0.25s; border-left-color: #55A868; }
.kpi-card:nth-child(4) { animation-delay: 0.35s; border-left-color: #C44E52; }
@keyframes cardFadeUp {
    from { opacity: 0; transform: translateY(16px); }
    to   { opacity: 1; transform: translateY(0); }
}
.kpi-label { font-size: 0.78rem; color: #666; font-weight: 600; text-transform: uppercase; letter-spacing: 0.04em; }
.kpi-value { font-size: 1.8rem; font-weight: 700; color: #222; margin-top: 4px; }

/* Tabs: smoother hover + underline transition */
.stTabs [data-baseweb="tab"] { transition: color 0.25s ease; }
.stTabs [data-baseweb="tab"]:hover { color: #4C72B0; }
.stTabs [data-baseweb="tab-highlight"] { transition: left 0.3s ease, width 0.3s ease; }

/* Chart hover zoom */
[data-testid="stImage"] img { transition: transform 0.3s ease; border-radius: 10px; }
[data-testid="stImage"] img:hover { transform: scale(1.015); }

/* Dataframe hover lift */
[data-testid="stDataFrame"] { transition: box-shadow 0.3s ease; border-radius: 10px; }
[data-testid="stDataFrame"]:hover { box-shadow: 0 6px 18px rgba(0,0,0,0.12); }

/* Custom spinner colour to match theme */
.stSpinner > div > div { border-top-color: #4C72B0 !important; }

/* Buttons: subtle lift on hover */
.stButton > button, .stDownloadButton > button {
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.stButton > button:hover, .stDownloadButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 14px rgba(76,114,176,0.25);
}
</style>
""", unsafe_allow_html=True)


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


with st.spinner("🏨 Loading and preparing booking data..."):
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
st.markdown("""
<div class="dash-header">
    <h1>🏨 Investigate Hotel Business — Booking &amp; Cancellation Dashboard</h1>
    <p>Data Visualization Project · Hotel bookings 2017–2019 · Cleaned dataset</p>
</div>
""", unsafe_allow_html=True)

kpis = [
    {"label": "Total Bookings", "value": len(filtered), "decimals": 0, "prefix": "", "suffix": ""},
    {"label": "Cancellation Rate", "value": filtered["is_canceled"].mean() * 100, "decimals": 1, "prefix": "", "suffix": "%"},
    {"label": "Avg. Lead Time", "value": filtered["lead_time"].mean(), "decimals": 0, "prefix": "", "suffix": " days"},
    {"label": "Avg. Daily Rate (ADR)", "value": filtered["adr"].mean(), "decimals": 2, "prefix": "$", "suffix": ""},
]

cards_html = '<div class="kpi-row">'
for k in kpis:
    cards_html += f"""
    <div class="kpi-card">
        <div class="kpi-label">{k['label']}</div>
        <div class="kpi-value">
            <span class="counter" data-target="{k['value']:.4f}" data-decimals="{k['decimals']}"
                  data-prefix="{k['prefix']}" data-suffix="{k['suffix']}">0</span>
        </div>
    </div>"""
cards_html += "</div>"
st.markdown(cards_html, unsafe_allow_html=True)

# Count-up animation script. Streamlit renders components.html in an iframe,
# so we reach into the parent document to animate the KPI numbers that were
# just written above via st.markdown.
components.html("""
<script>
setTimeout(function() {
    const counters = window.parent.document.querySelectorAll('.counter');
    counters.forEach(function(counter) {
        if (counter.dataset.animated === "1") return;
        counter.dataset.animated = "1";
        const target = parseFloat(counter.getAttribute('data-target'));
        const decimals = parseInt(counter.getAttribute('data-decimals'));
        const prefix = counter.getAttribute('data-prefix') || '';
        const suffix = counter.getAttribute('data-suffix') || '';
        const duration = 900;
        const startTime = performance.now();
        function animate(time) {
            const progress = Math.min((time - startTime) / duration, 1);
            const eased = 1 - Math.pow(1 - progress, 3);
            const current = target * eased;
            counter.textContent = prefix + current.toLocaleString('en-US', {
                minimumFractionDigits: decimals, maximumFractionDigits: decimals
            }) + suffix;
            if (progress < 1) requestAnimationFrame(animate);
        }
        requestAnimationFrame(animate);
    });
}, 50);
</script>
""", height=0)

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