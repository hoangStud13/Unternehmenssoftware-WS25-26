import matplotlib.pyplot as plt
import pandas as pd
import os

# --- CSV laden ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
data_dir = os.path.join(project_root, 'data')
features_csv = os.path.join(data_dir, 'nasdaq100_index_1m_features.csv')

script_dir = os.path.dirname(__file__)  # Ordner, in dem das Script liegt
images_dir = os.path.join(script_dir, "..", "..", "images")
images_dir = os.path.abspath(images_dir)


df = pd.read_csv(features_csv)

# Timestamps korrekt parsen
df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

# Features
cols = ["ema_5", "ema_20"]

# --- Tag auswählen ---
day = "2020-12-04"
day_dt = pd.to_datetime(day).date()

df_day = df[df["timestamp"].dt.date == day_dt]

if df_day.empty:
    raise ValueError(
        f"Keine Daten für {day}! "
        f"Verfügbare Tage: {sorted(df['timestamp'].dt.date.unique())}"
    )

# --- Zeitfilter 06:00–22:00 ---
df_day = df_day[
    (df_day["timestamp"].dt.hour >= 6) &
    (df_day["timestamp"].dt.hour <= 22)
]

# --- Stündlich aggregieren ---
df_hourly = df_day.set_index("timestamp")[cols].resample("1H").mean()

# --- Plot ---
plt.figure(figsize=(16, 8))

for c in cols:
    plt.plot(df_hourly.index, df_hourly[c], label=c)

plt.title(f"Stündliche EMAs für {day} (06:00–22:00)")
plt.xlabel("Uhrzeit")
plt.ylabel("Werte")
plt.grid(True)
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f"{images_dir}/03_emas.png")


# Features
cols = ["simple_return_1m", "simple_return_5m","simple_return_15m"]

# --- Tag auswählen ---
day = "2020-12-04"
day_dt = pd.to_datetime(day).date()

df_day = df[df["timestamp"].dt.date == day_dt]

if df_day.empty:
    raise ValueError(
        f"Keine Daten für {day}! "
        f"Verfügbare Tage: {sorted(df['timestamp'].dt.date.unique())}"
    )

# --- Zeitfilter 06:00–22:00 ---
df_day = df_day[
    (df_day["timestamp"].dt.hour >= 6) &
    (df_day["timestamp"].dt.hour <= 22)
]

# --- Stündlich aggregieren ---
df_hourly = df_day.set_index("timestamp")[cols].resample("1h").mean()

# --- Plot ---
plt.figure(figsize=(16, 8))

for c in cols:
    plt.plot(df_hourly.index, df_hourly[c], label=c)

plt.title(f"Stündliche returns")
plt.xlabel("Uhrzeit")
plt.ylabel("Werte")
plt.grid(True)
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f"{images_dir}/03_returns.png")
