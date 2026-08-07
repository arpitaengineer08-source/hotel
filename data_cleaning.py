

import pandas as pd
import numpy as np
import os

RAW_PATH = "hotel_bookings_data.csv"
CLEAN_PATH = os.path.join("data", "hotel_bookings_clean.csv")


def load_raw_data(path: str = RAW_PATH) -> pd.DataFrame:
    """Load the raw CSV exactly as provided."""
    df = pd.read_csv(path)
    return df


def data_overview(df: pd.DataFrame) -> None:
   
    print(f"Rows: {df.shape[0]:,} | Columns: {df.shape[1]}")
    print(f"Years covered: {sorted(df['arrival_date_year'].unique())}")
    print(f"Missing values by column:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
    print(f"Duplicate rows: {df.duplicated().sum():,}")


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
   
    df = df.copy()

    # 1. children
    df["children"] = df["children"].fillna(0)

    # 2. city
    df["city"] = df["city"].fillna("Unknown")

    # 3. agent
    df["agent"] = df["agent"].fillna(0)

    # 4. company
    df["company"] = df["company"].fillna(0)

    # 5. meal: recode 'Undefined' -> 'No Meal'
    df["meal"] = df["meal"].replace("Undefined", "No Meal")

    # 6. duplicates
    before = len(df)
    df = df.drop_duplicates()
    print(f"Dropped {before - len(df):,} duplicate rows.")

    # 7a. negative adr
    before = len(df)
    df = df[df["adr"] >= 0]
    print(f"Dropped {before - len(df):,} rows with negative adr.")

    # 7b. extreme adr outliers
    before = len(df)
    df = df[df["adr"] <= 5000]
    print(f"Dropped {before - len(df):,} rows with adr > 5000.")

    # 7c. zero total guests
    before = len(df)
    total_guests = df["adults"] + df["children"] + df["babies"]
    df = df[total_guests > 0]
    print(f"Dropped {before - len(df):,} rows with zero total guests.")

    # ---- Derived / helper columns used throughout the analysis ----
    df["total_stay_nights"] = df["stays_in_weekend_nights"] + df["stays_in_weekdays_nights"]
    df["total_guests"] = df["adults"] + df["children"] + df["babies"]
    df["is_canceled_label"] = df["is_canceled"].map({0: "Not Canceled", 1: "Canceled"})

    month_order = [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December",
    ]
    df["arrival_date_month"] = pd.Categorical(
        df["arrival_date_month"], categories=month_order, ordered=True
    )

    # Lead time buckets for question 3 (used by both the EDA script and dashboard)
    bins = [-1, 7, 30, 90, 180, 365, np.inf]
    labels = ["0-7 days", "8-30 days", "31-90 days", "91-180 days", "181-365 days", "365+ days"]
    df["lead_time_bucket"] = pd.cut(df["lead_time"], bins=bins, labels=labels)

    # Stay-duration buckets for question 2
    stay_bins = [-1, 1, 3, 7, 14, np.inf]
    stay_labels = ["1 night", "2-3 nights", "4-7 nights", "8-14 nights", "15+ nights"]
    df["stay_duration_bucket"] = pd.cut(df["total_stay_nights"], bins=stay_bins, labels=stay_labels)

    df = df.reset_index(drop=True)
    return df


def main():
    df_raw = load_raw_data()
    print("=== RAW DATA OVERVIEW ===")
    data_overview(df_raw)

    print("\n=== CLEANING ===")
    df_clean = clean_data(df_raw)

    print("\n=== CLEAN DATA OVERVIEW ===")
    print(f"Rows: {df_clean.shape[0]:,} | Columns: {df_clean.shape[1]}")
    print(f"Remaining missing values:\n{df_clean.isnull().sum()[df_clean.isnull().sum() > 0]}")

    os.makedirs("data", exist_ok=True)
    df_clean.to_csv(CLEAN_PATH, index=False)
    print(f"\nSaved cleaned dataset -> {CLEAN_PATH}")


if __name__ == "__main__":
    main()
