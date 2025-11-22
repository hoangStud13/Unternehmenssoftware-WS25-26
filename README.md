# NASDAQ Trading Bot mit Machine Learning

## Problem Definition:

Wir möchten für QQQ vorhersagen, wie sich der Preis in den nächsten
1, 3, 5, 10 und 15 Minuten bewegt.
Das Ziel ist eine kurzfristige Daytrading-Prognose, mit der man Trends früh erkennt.

Wir nutzen dafür Minuten-Daten und News, um Muster wie Momentum, Volatilität oder News-Impulse zu erkennen.

### Ziel

Ein Modell zu entwickeln, das:

- Preis- und Volumenmuster erkennt

- News-Einfluss kurzfristig berücksichtigt

- zuverlässig abschätzt, wohin sich QQQ in den nächsten Minuten bewegt

Damit können wir präzisere Daytrading-Signale erzeugen.

### Input   

Wir verwenden 1-Minuten-Bars:

- `timestamp`

- `open`

- `high`

- `low`

- `close`

- `volume`

- `trade_count`

- `vwap`

Zusätzlich nutzen wir News:

- `news_time`

- `sentiment`

### Input Features

#### Preis- und Trendfeatures

- 1-, 5- und 15-Minuten Returns

- EMA(5) und EMA(20)

- Unterschied zwischen kurzen und langen EMAs

- Realisierte Volatilität der letzten 10 Minuten

- High-Low-Range der aktuellen Minute

#### Volumen-Features

- Volumen-Z-Score über die letzten 30 Minuten

- Volumen pro Trade

#### News-Feature

- Effektives News-Sentiment (letzte News, abgeschwächt je älter sie ist)

### Output

Wir erstellen für jede Minute fünf Zielwerte:

- erwartete Preisänderung in 1 Minute

- erwartete Preisänderung in 3 Minuten

- erwartete Preisänderung in 5 Minuten

- erwartete Preisänderung in 10 Minuten

- erwartete Preisänderung in 15 Minuten
---

## 1 - Data Acquisition

Ruft historische 1 Minuten-Kerzendaten für einen NASDAQ-100 (QQQ) von 2020-11-23 bis 2025-11-20 ab. Die erfassten Markdaten werden sowohl im Parquet-Format als auch CSV-Format unter `nasdaq_trading_bot/data` gespeichert. 
Die Daten werden mit fester Endzeit geladen, um reproduzierbare Ergebnisse zu gewährleisten.

### Skript - Nasdaq Bar Fetcher

[scripts/01_data_acquisition/bar_retriever.py](nasdaq_trading_bot/scripts/01_data_acquisition/fetch_nasdaq_index.py)

### Bar-Daten Beispiel


| timestamp                 | open    | high    | low     | close   | volume  | trade_count | vwap    |
|---------------------------|--------|--------|--------|--------|--------|-------------|--------|
| 2020-11-23 09:00:00+00:00 | 282.40 | 282.40 | 282.39 | 282.40 | 2250.0 | 10.0        | 282.40 |
| 2020-11-23 09:01:00+00:00 | 282.38 | 282.38 | 282.33 | 282.35 | 1512.0 | 9.0         | 282.36 |
| 2020-11-23 09:02:00+00:00 | 282.39 | 282.39 | 282.33 | 282.33 | 437.0  | 6.0         | 282.36 |
| 2020-11-23 09:03:00+00:00 | 282.38 | 282.38 | 282.38 | 282.38 | 1203.0 | 6.0         | 282.38 |
| 2020-11-23 09:04:00+00:00 | 282.31 | 282.39 | 282.31 | 282.39 | 1184.0 | 4.0         | 282.33 |
| 2020-11-23 09:05:00+00:00 | 282.55 | 282.55 | 282.53 | 282.53 | 1185.0 | 6.0         | 282.53 |
| 2020-11-23 09:06:00+00:00 | 282.54 | 282.58 | 282.54 | 282.58 | 1147.0 | 8.0         | 282.54 |
| 2020-11-23 09:07:00+00:00 | 282.47 | 282.49 | 282.47 | 282.49 | 2600.0 | 6.0         | 282.47 |
| 2020-11-23 09:08:00+00:00 | 282.59 | 282.59 | 282.59 | 282.59 | 210.0  | 2.0         | 282.59 |
| 2020-11-23 09:10:00+00:00 | 282.57 | 282.61 | 282.57 | 282.61 | 900.0  | 4.0         | 282.60 |



### Skript - Nasdaq News Fetcher

ARMAN WÄSCHT SICH DIE HÄNDE NICHT NACHDEM ER AUF KLO WAR.

### News-Daten Beispiel

### 1.1 API - Dokumentation

#### NASDAQ Bar Fetcher (Alpaca API)
Ruft historische 1-Minuten-Kerzendaten für NASDAQ-100 (QQQ) über die Alpaca Data API ab. Die Daten werden im Parquet- und CSV-Format unter `nasdaq_trading_bot/data` gespeichert.

##### API Referenz

- **Endpoint**: `https://data.alpaca.markets/v2/stocks/bars`
- **Dokumentation**: [Alpaca Market Data API Docs](https://docs.alpaca.markets/api-documentation/api-v2/market-data/stocks/bars/)

##### Funktionen
- **Datenquelle**: Alpaca Market Data API (v2)
- **Zeitraum**: 2020-11-23 bis 2025-11-20
- **Symbol**: QQQ (Invesco QQQ Trust), NDX (Nasdaq-100 Index)
- **Zeitraum**: 1 Minute
- **Kursanpassung**: Alle Anpassungen (Splits, Dividenden)
- **Ausgabe**: Parquet- und CSV-Dateien mit OHLCV-Daten
- **Paginierung**: Automatische Handhabung der API-Paginierung für große Datensätze

##### Request Parameter

| Parameter         | Typ       | Beschreibung                            |
| ----------------- | --------- |-----------------------------------------|
| `symbol_or_symbols`| string    | Abgefragtes Symbol, z. B. `"QQQ"`       |
| `timeframe`       | TimeFrame | Zeitintervall, z. B. `TimeFrame.Minute` |
| `adjustment`      | enum      | Kursanpassung (`Adjustment.ALL`)        |
| `start`           | datetime  | Startzeitpunkt  (`START_DATE`)          |
| `end`             | datetime  | Endzeitpunkt          (`END_DATE`)      |

##### Bar-Datenstruktur - Spaltenbeschreibung

| Spalte        | Beschreibung                                                                 |
|---------------|-------------------------------------------------------------------------------|
| `timestamp`   | Zeitstempel der Kerze (z. B. 1min), inkl. Zeitzone (ISO-Format).             |
| `open`        | Eröffnungspreis der Periode.                                                  |
| `high`        | Höchster Preis der Periode.                                                   |
| `low`         | Tiefster Preis der Periode.                                                   |
| `close`       | Schlusskurs der Periode.                                                      |
| `volume`      | Gehandeltes Volumen innerhalb der Periode.                                    |
| `trade_count` | Anzahl der Trades in dieser Periode.                                          |
| `vwap`        | Volume Weighted Average Price – volumengewichteter Durchschnittspreis.        |

#### NASDAQ News Fetcher (Alpaca API)

Ruft historische Nachrichtenartikel für NASDAQ-bezogene Symbole (QQQ, NDX) über die Alpaca Data API ab. Es ist darauf ausgelegt, Daten der letzten 5 Jahre zu sammeln und in einer CSV-Datei unter `nasdaq_trading_bot/data` zu speichern.

##### API Referenz

- **Endpoint**: `GET /v1beta1/news`
- **Dokumentation**: [Alpaca News API Docs](https://docs.alpaca.markets/reference/news-1)


##### Funktionen

- **Datenquelle**: Alpaca News API (v1beta1)
- **Zeitraum**: Letzte 5 Jahre (dynamisch berechnet ab dem aktuellen Datum)
- **Symbole**: QQQ (Invesco QQQ Trust), NDX (Nasdaq-100 Index)
- **Ausgabe**: CSV-Datei mit relevanten Metadaten (Schlagzeile, Zusammenfassung, Autor, Zeitstempel, URL)
- **Bereinigung**: Entfernen der sympol-Spalte
- **Timezone**: US/Eastern

##### Request Parameter

| Parameter | Standardwert       | Beschreibung                                |
| --------- | ------------------ | ------------------------------------------- |
| `start`   | 5 Jahre vor Enddatum | Startdatum für die Datenabfrage             |
| `end`     | Aktuelles Datum    | Enddatum für die Datenabfrage               |
| `symbols` | QQQ, NDX           | Zu ladende Symbole / Assets                 |
| `limit`   | 50                 | Anzahl der Artikel/Bars pro Anfrage         |
| `sort`    | DESC               | Sortierreihenfolge (neueste zuerst) |


##### News-Datenstruktur - Spaltenbeschreibung

| Spalte       | Beschreibung                                                                 |
| :----------- | :--------------------------------------------------------------------------- |
| `id`         | Eindeutige ID des Artikels                                                   |
| `headline`   | Titel des Nachrichtenartikels                                                |
| `summary`    | Kurze Zusammenfassung des Inhalts                                            |
| `author`     | Autor oder Quelle des Artikels                                               |
| `created_at` | Erstellungsdatum und -uhrzeit (ISO 8601 Format)                              |
| `updated_at` | Datum und Uhrzeit der letzten Aktualisierung                                 |
| `url`        | Link zum vollständigen Artikel                                               |
| `symbols`    | Liste der zugehörigen Tickersymbole                                          |





---

## 2 - Data Understanding








