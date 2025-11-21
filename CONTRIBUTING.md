# Contributing Guide

Willkommen zum Trading Bot Projekt! Dieses Dokument enthält alle wichtigen Informationen für die Zusammenarbeit in unserem Team.

## Überblick

Wir entwickeln einen Trading Bot in Python als Uniprojekt. Dieses Repository ist privat und nur für eingeladene Teammitglieder zugänglich.

## Voraussetzungen

- Python 3.9 oder höher
- Git
- pip für Package-Management
- Grundkenntnisse in Python und Trading-Konzepten

## Setup

### 1. Repository klonen

```bash
git clone https://github.com/[username]/trading-bot.git
cd trading-bot
```

### 2. Virtuelle Umgebung erstellen

```bash
python -m venv venv
source venv/bin/activate  # Auf Windows: venv\Scripts\activate
```

### 3. Dependencies installieren

```bash
pip install -r requirements.txt
```

## Workflow

### Branches

Wir arbeiten mit folgender Branch-Struktur:

- `main` - Produktionsreifer, stabiler Code
- `develop` - Entwicklungsbranch für Integration
- `feature/*` - Neue Features (z.B. `feature/order-execution`)
- `bugfix/*` - Bugfixes (z.B. `bugfix/api-connection`)
- `docs/*` - Dokumentationsänderungen

### Einen neuen Branch erstellen

```bash
git checkout develop
git pull origin develop
git checkout -b feature/dein-feature-name
```

### Commits

Wir verwenden aussagekräftige Commit-Messages nach folgendem Schema:

```
<typ>: <kurze Beschreibung>

<optional: ausführliche Beschreibung>
```

**Typen:**
- `feat`: Neues Feature
- `fix`: Bugfix
- `docs`: Dokumentationsänderung
- `style`: Code-Formatierung (keine funktionalen Änderungen)
- `refactor`: Code-Refactoring
- `test`: Tests hinzufügen oder ändern
- `chore`: Build-Prozess oder Tool-Konfiguration

**Beispiele:**
```bash
git commit -m "feat: API-Integration für Binance hinzugefügt"
git commit -m "fix: Fehler bei der Orderausführung behoben"
git commit -m "docs: README mit Installationsanleitung aktualisiert"
```

### Pull Requests

1. Stelle sicher, dass dein Code funktioniert und getestet ist
2. Pushe deinen Branch:
   ```bash
   git push origin feature/dein-feature-name
   ```
3. Erstelle einen Pull Request auf GitHub von deinem Branch nach `develop`
4. Füge eine aussagekräftige Beschreibung hinzu
5. Markiere mindestens ein Teammitglied als Reviewer
6. Warte auf das Review und die Freigabe
7. Nach Approval: Merge in `develop`

## Code-Standards

### Python Style Guide

Wir folgen PEP 8. Wichtigste Punkte:

- Einrückung: 4 Leerzeichen
- Maximale Zeilenlänge: 100 Zeichen
- Funktions- und Variablennamen: `snake_case`
- Klassennamen: `PascalCase`
- Konstanten: `UPPER_CASE`

### Dokumentation

- Jede Funktion und Klasse benötigt einen Docstring
- Verwende Google-Style Docstrings:

```python
def calculate_profit(entry_price, exit_price, quantity):
    """
    Berechnet den Gewinn eines Trades.
    
    Args:
        entry_price (float): Einstiegspreis
        exit_price (float): Ausstiegspreis
        quantity (float): Anzahl der gehandelten Einheiten
        
    Returns:
        float: Gewinn in absoluten Zahlen
    """
    return (exit_price - entry_price) * quantity
```

### Tests

- Schreibe Unit-Tests für neue Funktionen
- Tests liegen im `tests/`-Verzeichnis
- Verwende pytest als Testing-Framework
- Führe Tests vor dem Push aus:
  ```bash
  pytest tests/
  ```

## Projektstruktur

```
trading-bot/
├── src/
│   ├── api/          # API-Integrationen
│   ├── strategies/   # Trading-Strategien
│   ├── utils/        # Hilfsfunktionen
│   └── main.py       # Haupteinstiegspunkt
├── tests/            # Unit-Tests
├── docs/             # Dokumentation
├── requirements.txt  # Python-Dependencies
└── README.md         # Projektübersicht
```

## Wichtige Hinweise

### API-Keys und Secrets

- **NIEMALS** API-Keys oder Passwörter im Code commiten
- Verwende `.env`-Dateien für lokale Konfiguration
- Die `.env`-Datei ist in `.gitignore` aufgenommen
- Verwende `python-dotenv` zum Laden von Umgebungsvariablen

### Code Review

Beim Review achten wir auf:

- Funktionalität und Korrektheit
- Code-Qualität und Lesbarkeit
- Vorhandensein von Tests
- Dokumentation
- Einhaltung der Code-Standards

## Kommunikation

- Nutze Issues für Bugs und Feature-Requests
- Bei Fragen oder Problemen: Discord/Slack/E-Mail
- Regelmäßige Team-Meetings zur Abstimmung


