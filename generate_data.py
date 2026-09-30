"""Generates realistic synthetic energy consumption datasets for testing and verification."""

import os
import numpy as np
import pandas as pd


def generate_synthetic_data() -> None:
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(data_dir, exist_ok=True)
    np.random.seed(42)

    # 1. Dataset 1: Hourly - DateTime, Energy_Usage (30 days = 720 hours)
    dates1 = pd.date_range(start="2026-01-01 00:00:00", periods=720, freq="1h")
    hour1 = dates1.hour.to_numpy()
    dow1 = dates1.dayofweek.to_numpy()
    diurnal1 = 15.0 + 8.0 * np.sin(2 * np.pi * (hour1 - 6) / 24)
    weekend_effect1 = np.where(dow1 >= 5, -4.0, 0.0)
    noise1 = np.random.normal(0, 1.5, size=len(dates1))
    energy1 = np.maximum(2.0, diurnal1 + weekend_effect1 + noise1)
    df1 = pd.DataFrame({"DateTime": dates1.strftime("%Y-%m-%d %H:%M:%S"), "Energy_Usage": np.round(energy1, 2)})
    df1.to_csv(os.path.join(data_dir, "dataset1.csv"), index=False)

    # 2. Dataset 2: 30-min - timestamp, load (1000 steps = ~21 days)
    dates2 = pd.date_range(start="2026-02-01 00:00:00", periods=1000, freq="30min")
    hour2 = dates2.hour.to_numpy() + dates2.minute.to_numpy() / 60.0
    load2 = 120.0 + 40.0 * np.sin(2 * np.pi * (hour2 - 7) / 24) + np.random.normal(0, 5.0, size=len(dates2))
    # Inject a realistic energy spike
    load2[350] += 80.0
    df2 = pd.DataFrame({"timestamp": dates2.strftime("%Y-%m-%dT%H:%M:%S"), "load": np.round(load2, 2)})
    df2.to_csv(os.path.join(data_dir, "dataset2.csv"), index=False)

    # 3. Dataset 3: 15-min with Weather - reading_time, power_consumption, temperature, humidity (1500 steps)
    dates3 = pd.date_range(start="2026-03-01 00:00:00", periods=1500, freq="15min")
    hour3 = dates3.hour.to_numpy() + dates3.minute.to_numpy() / 60.0
    temp3 = 18.0 + 10.0 * np.sin(2 * np.pi * (hour3 - 9) / 24) + np.random.normal(0, 1.0, size=len(dates3))
    humidity3 = 60.0 - 20.0 * np.sin(2 * np.pi * (hour3 - 9) / 24) + np.random.normal(0, 3.0, size=len(dates3))
    power3 = 45.0 + 15.0 * np.sin(2 * np.pi * (hour3 - 6) / 24) + 0.8 * np.maximum(0, temp3 - 22.0) + np.random.normal(0, 2.0, size=len(dates3))
    df3 = pd.DataFrame({
        "reading_time": dates3.strftime("%m/%d/%Y %H:%M"),
        "power_consumption": np.round(power3, 2),
        "temperature": np.round(temp3, 1),
        "humidity": np.round(humidity3, 1),
    })
    df3.to_csv(os.path.join(data_dir, "dataset3.csv"), index=False)

    # 4. Dataset 4: Daily - date, electricity_demand (365 days)
    dates4 = pd.date_range(start="2025-01-01", periods=365, freq="1D")
    doy4 = dates4.dayofyear.to_numpy()
    demand4 = 500.0 + 120.0 * np.sin(2 * np.pi * (doy4 - 15) / 365) + np.random.normal(0, 20.0, size=len(dates4))
    df4 = pd.DataFrame({"date": dates4.strftime("%Y-%m-%d"), "electricity_demand": np.round(demand4, 1)})
    df4.to_csv(os.path.join(data_dir, "dataset4.csv"), index=False)

    # 5. Dataset 5: Hourly with Temp - time, value, temp (600 hours)
    dates5 = pd.date_range(start="2026-04-01 00:00:00", periods=600, freq="1h")
    hour5 = dates5.hour.to_numpy()
    temp5 = 15.0 + 8.0 * np.sin(2 * np.pi * (hour5 - 8) / 24) + np.random.normal(0, 1.2, size=len(dates5))
    val5 = 30.0 + 12.0 * np.sin(2 * np.pi * (hour5 - 5) / 24) + 0.5 * temp5 + np.random.normal(0, 1.8, size=len(dates5))
    df5 = pd.DataFrame({"time": dates5.strftime("%Y-%m-%d %H:%M"), "value": np.round(val5, 2), "temp": np.round(temp5, 1)})
    df5.to_csv(os.path.join(data_dir, "dataset5.csv"), index=False)

    # 6. Sample canonical energy dataset (720 hours)
    dates_sample = pd.date_range(start="2026-05-01 00:00:00", periods=720, freq="1h")
    hour_s = dates_sample.hour.to_numpy()
    temp_s = 20.0 + 7.0 * np.sin(2 * np.pi * (hour_s - 8) / 24) + np.random.normal(0, 0.8, size=len(dates_sample))
    hum_s = 55.0 - 15.0 * np.sin(2 * np.pi * (hour_s - 8) / 24) + np.random.normal(0, 2.0, size=len(dates_sample))
    cons_s = 50.0 + 20.0 * np.sin(2 * np.pi * (hour_s - 6) / 24) + 1.2 * np.maximum(0, temp_s - 22.0) + np.random.normal(0, 2.5, size=len(dates_sample))
    df_sample = pd.DataFrame({
        "timestamp": dates_sample.strftime("%Y-%m-%d %H:%M:%S"),
        "consumption": np.round(cons_s, 2),
        "temperature": np.round(temp_s, 1),
        "humidity": np.round(hum_s, 1),
    })
    df_sample.to_csv(os.path.join(data_dir, "sample_energy.csv"), index=False)

    # 7. Invalid dataset (Non-time-series, non-energy)
    df_inv = pd.DataFrame({
        "Name": ["Alice", "Bob", "Charlie", "Diana", "Ethan", "Fiona"],
        "Age": [24, 30, 22, 28, 35, 29],
        "Marks": [88.5, 92.0, 79.5, 95.0, 81.0, 89.0],
        "City": ["New York", "Chicago", "Boston", "Seattle", "Austin", "Denver"],
    })
    df_inv.to_csv(os.path.join(data_dir, "invalid.csv"), index=False)

    print("Sample datasets successfully created in:", data_dir)


if __name__ == "__main__":
    generate_synthetic_data()
