"""
data_fetcher.py — Fetch dati EODHD con caching Streamlit.

Una sola responsabilità: scaricare la serie storica giornaliera completa di un
asset e restituire una colonna di prezzo pulita e pronta per l'analisi. Tutte le
trasformazioni quantitative vivono in calculations.py.
"""

from __future__ import annotations

import requests
import pandas as pd
import streamlit as st

# Tutto lo storico disponibile: EODHD restituisce dalla quotazione/inizio serie.
DEFAULT_START = "1900-01-01"


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_eod(ticker: str, api_key: str,
              start: str = DEFAULT_START, end: str | None = None) -> pd.DataFrame:
    """Scarica e cacha l'intera serie EOD giornaliera da EODHD.

    Args:
        ticker:  Simbolo EODHD (es. 'GSPC.INDX', 'AAPL.US', 'BTC-USD.CC',
                 'EURUSD.FOREX').
        api_key: Chiave API EODHD.
        start:   Data inizio YYYY-MM-DD (default = tutto lo storico).
        end:     Data fine YYYY-MM-DD (default = oggi).

    Returns:
        DataFrame con DatetimeIndex ordinato e colonne OHLCV + adjusted_close.
        DataFrame vuoto se la risposta non contiene dati.

    Raises:
        requests.HTTPError per errori API (chiave non valida, ticker inesistente...).
    """
    end = end or pd.Timestamp.today().strftime("%Y-%m-%d")
    url = f"https://eodhd.com/api/eod/{ticker}"
    params = {
        "from": start,
        "to": end,
        "period": "d",
        "api_token": api_key,
        "fmt": "json",
    }
    resp = requests.get(url, params=params, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    cols = ["open", "high", "low", "close", "volume", "adjusted_close"]
    df = df[[c for c in cols if c in df.columns]].apply(pd.to_numeric, errors="coerce")
    return df


def get_price_series(df: pd.DataFrame) -> pd.Series:
    """Estrae la serie di prezzo da usare per l'analisi.

    Preferisce 'adjusted_close' (dividendi e split inclusi) e ripiega su 'close'
    quando l'adjusted non è disponibile (alcuni indici/forex). Rimuove i valori
    non positivi che renderebbero indefiniti i rendimenti.
    """
    col = "adjusted_close" if "adjusted_close" in df.columns else "close"
    price = df[col].dropna()
    return price[price > 0]
