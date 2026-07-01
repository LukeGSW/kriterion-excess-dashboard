# 📊 Analisi degli Eccessi — Kriterion Quant

Dashboard Streamlit per studiare, asset per asset, **cosa è storicamente accaduto
ai rendimenti dopo un eccesso** — cioè dopo che un periodo ha chiuso oltre N
deviazioni standard. Lo studio gira su tre orizzonti temporali:

| Orizzonte | Eccesso misurato su | Cosa guarda dopo |
|-----------|---------------------|------------------|
| Giornaliero | rendimento giornaliero | 5 giorni dopo |
| Settimanale | rendimento settimanale | 4 settimane dopo |
| Mensile | rendimento mensile | 3 mesi dopo |

L'obiettivo è capire se gli estremi vengono **riassorbiti** (mean reversion) o
**estesi** (momentum), e con quale **winrate** — sempre confrontato con una
baseline incondizionata per misurare l'*edge* reale.

---

## Come funziona lo studio

- **Eccesso.** Un periodo è un eccesso quando il suo rendimento supera la soglia
  `mediana ± N·σ`. La soglia può essere:
  - **Statica** — mediana e σ sull'intero campione (coincide con le bande
    dell'istogramma).
  - **Adattiva** (default) — z-score sulla volatilità recente, regime-aware:
    confronta in modo equo epoche con volatilità molto diversa. Su storici corti
    si comporta come *expanding* per non perdere dati.
- **Win direzionale.** Dopo un eccesso SOPRA il win è un rendimento forward
  negativo (reversione ↓); dopo un eccesso SOTTO, positivo (reversione ↑).
- **Cooldown = orizzonte.** Dopo ogni evento si saltano K periodi, così le
  finestre forward non si sovrappongono: i campioni restano indipendenti e il
  winrate non è gonfiato dall'autocorrelazione.
- **Edge.** `Edge = winrate condizionato − baseline incondizionata`. È il numero
  che conta davvero.
- **Affidabilità.** 🟢 ≥ 30 eventi · 🟡 10–29 · 🔴 < 10. I risultati fragili non
  vengono nascosti, ma marcati.

---

## Struttura

```
kriterion-excess-dashboard/
├── app.py                  # Entry point Streamlit (layout e rendering)
├── requirements.txt
├── .streamlit/
│   ├── config.toml         # Tema scuro
│   └── secrets.toml.example
├── src/
│   ├── data_fetcher.py     # Fetch EODHD + caching
│   ├── calculations.py     # Logica quantitativa pura (eventi, cooldown, metriche)
│   └── charts.py           # Grafici Plotly (istogramma eccessi, forward)
└── README.md
```

---

## Esecuzione locale

1. Installa le dipendenze:
   ```bash
   pip install -r requirements.txt
   ```
2. Crea `.streamlit/secrets.toml` (copia da `secrets.toml.example`) con la tua
   chiave EODHD:
   ```toml
   EODHD_API_KEY = "la-tua-chiave-eodhd"
   ```
3. Avvia:
   ```bash
   streamlit run app.py
   ```

---

## Deploy su Streamlit Cloud

1. Push del repository su GitHub.
2. [streamlit.io/cloud](https://streamlit.io/cloud) → **New app** → seleziona il repo.
3. **Advanced settings → Secrets**, incolla:
   ```toml
   EODHD_API_KEY = "la-tua-chiave-eodhd"
   ```
4. **Deploy**.

---

## Formato ticker (EODHD)

| Asset | Esempio |
|-------|---------|
| Indice | `GSPC.INDX` (S&P 500), `DJI.INDX` |
| Azione US | `AAPL.US`, `TSLA.US` |
| Azione IT | `ENI.MI` |
| Crypto | `BTC-USD.CC`, `ETH-USD.CC` |
| Forex | `EURUSD.FOREX` |

---

## Avvertenze

Studio **descrittivo** su dati storici, non un sistema di trading. Volatility
clustering, cambi di regime e costi di transazione non sono modellati. Performance
passate non garantiscono risultati futuri.
