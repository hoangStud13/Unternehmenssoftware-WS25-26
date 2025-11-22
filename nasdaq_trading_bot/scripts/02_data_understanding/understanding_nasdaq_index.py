import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# =========================== #
# SETUP                       #
# =========================== #
# OS - Holt Ordnerstruktur
script_dir = os.path.dirname(__file__)  # Ordner, in dem das Script liegt
data_path = os.path.join(script_dir, "..", "..", "data", "nasdaq100_index_1m.csv")
data_path = os.path.abspath(data_path)

images_dir = os.path.join(script_dir, "..", "..", "images")
images_dir = os.path.abspath(images_dir)


# =========================== #
# Analyse Start               #
# =========================== #

"""
Steps:
Überblick bekommen
    1. Header anzeigen
    2. Values, Datentypen, non-null
    3. Statistische Kennzahlen
    4. Erste 10 Zeilen
    5. Letzte 10 Zeilen

Datenqualität prüfen
    1. Null Werte
    2. Doppelte Werte
    
Verteilung analysieren
    1. Histogram (Verteilung der Daten) - Eher wenig Sinn, da Zeitbasiert
    2. Liniendiagramm (Preisentwicklung der Daten)


"""

data = pd.read_csv(data_path)


print("Übersicht Header")
print(data.info())


print("Statistische Kennzahlen")
print(data.describe())


print(data.head(10))


print("Letzte 10 Zeilen")
print(data.tail(10))

# Datenqualität prüfen
print("Doppelte Werte:")
print(data.duplicated().sum())

print("Null Werte:")
print(data.isnull().sum())


# Verteilung analysieren

# Verteilung der Open-Preise
plt.hist(data["open"],bins=30, color="blue",edgecolor="black")
plt.title("Verteilung der Open-Preise")
plt.xlabel("Preis in USD")
plt.ylabel("Anzahl der Tage")
plt.savefig(f"{images_dir}/02_Verteilung_analysieren.png")


# Wöchentliche Open-Preise
line_chart_df = data[['timestamp', 'open']].copy()
line_chart_df['timestamp'] = pd.to_datetime(line_chart_df['timestamp'])
line_chart_df.set_index('timestamp', inplace=True)


line_chart_df_weekly = line_chart_df['open'].resample('W').mean()

weekly_returns = line_chart_df_weekly.pct_change().dropna()

plt.figure(figsize=(12,6))
plt.plot(line_chart_df_weekly.index, line_chart_df_weekly.values, label='Wöchentlicher Open-Preis', color='blue')
plt.title("Wöchentliche Open-Preise")
plt.xlabel("Datum")
plt.ylabel("Open-Preis")
plt.grid(True)
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f"{images_dir}/02_weekly_open_price.png")

# Moving Average 50 Tage und 200 Tage
mv50 = data.copy()
mv50['timestamp'] = pd.to_datetime(mv50['timestamp'])
mv50.set_index('timestamp', inplace=True)

# Auf Tagesbasis aggregieren (Durchschnitt Open-Preis pro Tag)
ma_daily = mv50['open'].resample('D').mean()

# NaN-Werte entfernen (Wochenenden ohne Handel)
ma_daily = ma_daily.dropna()

# 50-Tage und 200-Tage gleitender Durchschnitt
ma50 = ma_daily.rolling(window=50).mean()
ma200 = ma_daily.rolling(window=200).mean()


# Plotten - ALLE Daten, auch mit NaN am Anfang
plt.figure(figsize=(12,6))
plt.plot(ma_daily.index, ma_daily.values,
         label='Täglicher Open-Preis', color='blue', linewidth=0.8, alpha=0.7)
plt.plot(ma50.index, ma50.values,
         label='50-Tage MA', color='orange', linewidth=2)
plt.plot(ma200.index, ma200.values,
         label='200-Tage MA', color='green', linewidth=2)
plt.title("Open-Preis mit gleitendem Durchschnitt")
plt.xlabel("Datum")
plt.ylabel("Preis in USD")
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f"{images_dir}/02_moving_average.png")




