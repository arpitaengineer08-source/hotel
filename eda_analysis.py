

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from data_cleaning import load_raw_data, clean_data, CLEAN_PATH

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 110
CHART_DIR = "charts"


def get_clean_df() -> pd.DataFrame:
    if os.path.exists(CLEAN_PATH):
        df = pd.read_csv(CLEAN_PATH)
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
    df_raw = load_raw_data()
    return clean_data(df_raw)


# ---------------------------------------------------------------------
# Q1: Which hotel type do customers book most often?
# ---------------------------------------------------------------------
def q1_hotel_type_share(df: pd.DataFrame):
    counts = df["hotel"].value_counts()
    pct = (counts / counts.sum() * 100).round(1)
    print("\n--- Q1: Hotel type share ---")
    print(counts)
    print(pct.astype(str) + "%")

    fig, ax = plt.subplots(figsize=(6, 5))
    colors = ["#4C72B0", "#DD8452"]
    ax.pie(counts, labels=counts.index, autopct="%1.1f%%", colors=colors, startangle=90)
    ax.set_title("Share of Bookings by Hotel Type")
    fig.text(0.5, -0.02, "Caption: City Hotel accounts for the majority of bookings.",
              ha="center", fontsize=9, style="italic")
    fig.tight_layout()
    fig.savefig(f"{CHART_DIR}/q1_hotel_type_share.png", bbox_inches="tight")
    plt.close(fig)

    monthly = df.groupby(["arrival_date_month", "hotel"], observed=True).size().reset_index(name="bookings")
    fig, ax = plt.subplots(figsize=(11, 5))
    sns.lineplot(data=monthly, x="arrival_date_month", y="bookings", hue="hotel", marker="o", ax=ax)
    ax.set_title("Monthly Bookings by Hotel Type")
    ax.set_xlabel("Month")
    ax.set_ylabel("Number of Bookings")
    ax.tick_params(axis="x", rotation=45)
    fig.text(0.5, -0.05, "Caption: Bookings peak in summer months (Jul-Aug) for both hotel types.",
              ha="center", fontsize=9, style="italic")
    fig.tight_layout()
    fig.savefig(f"{CHART_DIR}/q1_monthly_bookings.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------
# Q2: Does length of stay affect cancellation rate?
# ---------------------------------------------------------------------
def q2_stay_duration_vs_cancellation(df: pd.DataFrame):
    cancel_by_hotel = df.groupby("hotel")["is_canceled"].mean().round(3) * 100
    print("\n--- Q2: Overall cancellation rate by hotel type (%) ---")
    print(cancel_by_hotel)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.barplot(x=cancel_by_hotel.index, y=cancel_by_hotel.values, hue=cancel_by_hotel.index,
                palette=["#4C72B0", "#DD8452"], legend=False, ax=ax)
    ax.set_title("Overall Cancellation Rate by Hotel Type")
    ax.set_ylabel("Cancellation Rate (%)")
    ax.set_xlabel("")
    for i, v in enumerate(cancel_by_hotel.values):
        ax.text(i, v + 0.5, f"{v:.1f}%", ha="center")
    fig.text(0.5, -0.03, "Caption: City Hotel bookings are cancelled more often than Resort Hotel bookings.",
              ha="center", fontsize=9, style="italic")
    fig.tight_layout()
    fig.savefig(f"{CHART_DIR}/q2_cancellation_by_hotel.png", bbox_inches="tight")
    plt.close(fig)

    stay_cancel = (
        df.groupby(["stay_duration_bucket", "hotel"], observed=True)["is_canceled"]
        .mean().mul(100).reset_index(name="cancel_rate")
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(data=stay_cancel, x="stay_duration_bucket", y="cancel_rate", hue="hotel", ax=ax)
    ax.set_title("Cancellation Rate by Length of Stay")
    ax.set_xlabel("Total Stay Duration")
    ax.set_ylabel("Cancellation Rate (%)")
    fig.text(0.5, -0.05, "Caption: Cancellation rate generally rises with longer stays, more sharply for City Hotel.",
              ha="center", fontsize=9, style="italic")
    fig.tight_layout()
    fig.savefig(f"{CHART_DIR}/q2_cancellation_by_stay_duration.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------
# Q3: Does lead time affect cancellation rate?
# ---------------------------------------------------------------------
def q3_lead_time_vs_cancellation(df: pd.DataFrame):
    lead_cancel = (
        df.groupby(["lead_time_bucket", "hotel"], observed=True)["is_canceled"]
        .mean().mul(100).reset_index(name="cancel_rate")
    )
    print("\n--- Q3: Cancellation rate by lead time bucket (%) ---")
    print(lead_cancel)

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(data=lead_cancel, x="lead_time_bucket", y="cancel_rate", hue="hotel", marker="o", ax=ax)
    ax.set_title("Cancellation Rate by Lead Time")
    ax.set_xlabel("Lead Time Bucket")
    ax.set_ylabel("Cancellation Rate (%)")
    fig.text(0.5, -0.05, "Caption: Cancellation rate climbs as lead time grows, especially for City Hotel.",
              ha="center", fontsize=9, style="italic")
    fig.tight_layout()
    fig.savefig(f"{CHART_DIR}/q3_cancellation_by_lead_time.png", bbox_inches="tight")
    plt.close(fig)


def main():
    os.makedirs(CHART_DIR, exist_ok=True)
    df = get_clean_df()

    q1_hotel_type_share(df)
    q2_stay_duration_vs_cancellation(df)
    q3_lead_time_vs_cancellation(df)

    print(f"\nAll charts saved to ./{CHART_DIR}/")


if __name__ == "__main__":
    main()
