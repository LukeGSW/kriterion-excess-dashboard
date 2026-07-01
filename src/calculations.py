"""
calculations.py — Logica quantitativa pura dello studio sugli eccessi.

Nessuna dipendenza da Streamlit o da chiamate API: sono funzioni testabili in
isolamento che ricevono Series/DataFrame puliti e restituiscono risultati numerici.

Concetto dello studio
----------------------
Per ciascun orizzonte (giornaliero, settimanale, mensile) si analizza la
distribuzione storica dei rendimenti di periodo. Quando un rendimento "chiude"
oltre N deviazioni standard (un *eccesso*), si misura cosa è successo K periodi
dopo (5 giorni / 4 settimane / 3 mesi). L'obiettivo è capire se gli eccessi
vengono riassorbiti (mean reversion) o estesi (momentum), e con quale winrate.

Scelte metodologiche (concordate in fase di design)
---------------------------------------------------
- Win DIREZIONALE: dopo un eccesso SOPRA il "win" è un rendimento forward
  negativo (reversione); dopo un eccesso SOTTO è un rendimento forward positivo.
- COOLDOWN = K periodi: dopo ogni evento si saltano K periodi, così le finestre
  forward non si sovrappongono e i campioni sono (quasi) indipendenti. Questo
  rende il winrate statisticamente onesto invece che gonfiato da autocorrelazione.
- Soglia STATICA (mediana ± Nσ full-sample) oppure ADATTIVA (z-score su
  volatilità recente, regime-aware) — la statica è anche il riferimento visivo
  dell'istogramma, l'adattiva è il default per gli storici lunghi.
- BASELINE incondizionata: ogni winrate è confrontato con la probabilità di
  "win" calcolata su TUTTI i periodi. L'edge (condizionato − baseline) è ciò
  che conta davvero, non il winrate assoluto.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

# === Configurazione orizzonti ============================================
# rule:        offset pandas per il resample (D = nessun resample)
# fwd:         numero di periodi forward da analizzare dopo l'eccesso
# roll_win/min finestra e min_periods per la deviazione standard adattiva
# sigma_levels bande di deviazione standard mostrate sull'istogramma
HORIZONS: dict[str, dict] = {
    "Giornaliero": dict(rule="D",     fwd=5, fwd_label="5 giorni",
                        roll_win=252, roll_min=60, sigma_levels=(1, 2, 3)),
    "Settimanale": dict(rule="W-FRI", fwd=4, fwd_label="4 settimane",
                        roll_win=104, roll_min=26, sigma_levels=(1, 2)),
    "Mensile":     dict(rule="ME",    fwd=3, fwd_label="3 mesi",
                        roll_win=36,  roll_min=12, sigma_levels=(1, 2)),
}

# Soglie di affidabilità sul numero di eventi indipendenti
REL_GREEN = 30    # >= 30 eventi -> statisticamente utilizzabile
REL_YELLOW = 10   # 10-29 eventi -> indicativo

# Livelli di N usati nella mini-tabella di sensibilità
SENSITIVITY_LEVELS = (1.5, 2.0, 2.5)


def reliability(n: int) -> tuple[str, str]:
    """Restituisce (emoji, etichetta) di affidabilità in base al numero di eventi."""
    if n >= REL_GREEN:
        return "🟢", "Affidabile"
    if n >= REL_YELLOW:
        return "🟡", "Indicativo"
    return "🔴", "Aneddotico"


# === Trasformazioni di base ==============================================

def resample_close(daily_price: pd.Series, rule: str) -> pd.Series:
    """Riporta la serie prezzi giornaliera all'orizzonte richiesto.

    Args:
        daily_price: Serie prezzi (adjusted close) con index giornaliero.
        rule:        'D' (nessun resample), 'W-FRI' o 'ME'.

    Returns:
        Serie prezzi di chiusura di periodo, NaN rimossi.
    """
    if rule == "D":
        return daily_price.dropna()
    return daily_price.resample(rule).last().dropna()


def period_returns(price: pd.Series) -> pd.Series:
    """Rendimenti semplici di periodo (pct_change), NaN rimossi."""
    return price.pct_change().dropna()


def build_return_frame(price: pd.Series, n_std: float, mode: str,
                       cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Costruisce il frame dei rendimenti con soglie ed eventi grezzi.

    Args:
        price:  Serie prezzi di periodo (chiusura).
        n_std:  Numero di deviazioni standard che definisce l'eccesso.
        mode:   'Statica' (mediana ± Nσ full-sample) o 'Adattiva'
                (z-score su volatilità recente, regime-aware).
        cfg:    Configurazione orizzonte (per roll_win/roll_min).

    Returns:
        (df_full, df_valid)
        df_full:  tutte le righe, con colonna 'price' e 'pos' (posizione intera
                  nella serie prezzi) — serve per le finestre forward e lo stato attuale.
        df_valid: solo righe con center/sigma definiti — usato per gli eventi.

    Colonne:
        price, ret, pos, center, sigma, z, up (bool), down (bool)
    """
    price = price.dropna()
    ret = price.pct_change()
    df = pd.DataFrame({"price": price.values, "ret": ret.values}, index=price.index)
    df["pos"] = np.arange(len(df))

    if mode == "Statica":
        # Mediana e deviazione standard sull'intero campione (descrittivo).
        df["center"] = ret.median()
        df["sigma"] = ret.std()
    else:
        # Adattiva: statistiche su volatilità recente. shift(1) -> la soglia al
        # tempo t usa solo informazione fino a t-1 (no look-ahead e la barra
        # estrema non gonfia la propria deviazione standard). min_periods piccolo
        # garantisce comportamento "expanding" all'inizio della serie, così non
        # si butta via storia preziosa sugli asset con poco storico.
        roll = df["ret"].rolling(cfg["roll_win"], min_periods=cfg["roll_min"])
        df["center"] = roll.median().shift(1)
        df["sigma"] = roll.std().shift(1)

    df["z"] = (df["ret"] - df["center"]) / df["sigma"]
    df["up"] = df["ret"] >= (df["center"] + n_std * df["sigma"])
    df["down"] = df["ret"] <= (df["center"] - n_std * df["sigma"])

    df_valid = df.dropna(subset=["ret", "center", "sigma"]).copy()
    return df, df_valid


# === Eventi, cooldown, finestre forward ==================================

def apply_cooldown(event_positions: np.ndarray, fwd: int) -> list[int]:
    """De-clustering: tiene solo eventi distanti almeno `fwd` periodi.

    Dopo aver accettato un evento alla posizione p, il prossimo evento ammesso
    deve trovarsi a p + fwd o oltre. Questo garantisce che le finestre forward
    non si sovrappongano (campioni indipendenti) ed evita di contare lo stesso
    eccesso più volte quando si presenta in barre consecutive.
    """
    kept: list[int] = []
    next_allowed = -1
    for pos in np.sort(event_positions):
        if pos >= next_allowed:
            kept.append(int(pos))
            next_allowed = pos + fwd
    return kept


def forward_returns(price_vals: np.ndarray, positions: list[int], k: int) -> np.ndarray:
    """Rendimento forward cumulato a K periodi per ogni posizione evento.

    Forward return = price[pos + k] / price[pos] - 1. Le posizioni la cui
    finestra forward esce dalla serie (pos + k oltre l'ultimo dato) sono scartate.
    """
    out = []
    n = len(price_vals)
    for pos in positions:
        if pos + k < n:
            out.append(price_vals[pos + k] / price_vals[pos] - 1.0)
    return np.asarray(out, dtype=float)


def baseline_forward(price_vals: np.ndarray, k: int) -> dict:
    """Distribuzione forward INCONDIZIONATA su tutti i periodi.

    Serve da metro di paragone: un winrate condizionato va sempre confrontato
    con la probabilità di 'win' che si avrebbe entrando a caso.

    Returns:
        dict con p_up = P(fwd > 0), p_down = P(fwd < 0), mean, median, e l'array.
    """
    n = len(price_vals)
    if n <= k + 1:
        return dict(p_up=np.nan, p_down=np.nan, mean=np.nan, median=np.nan,
                    values=np.array([]))
    fwd = price_vals[k:] / price_vals[:-k] - 1.0
    fwd = fwd[np.isfinite(fwd)]
    return dict(
        p_up=float((fwd > 0).mean()),
        p_down=float((fwd < 0).mean()),
        mean=float(fwd.mean()),
        median=float(np.median(fwd)),
        values=fwd,
    )


# === Metriche per direzione =============================================

def direction_metrics(fwd: np.ndarray, baseline_win: float, direction: str) -> dict:
    """Calcola le metriche dello studio per una direzione (up/down).

    Args:
        fwd:          rendimenti forward degli eventi (già de-clusterizzati).
        baseline_win: probabilità di win incondizionata per questa direzione.
        direction:    'up' (eccesso sopra, win = reversione giù) o
                      'down' (eccesso sotto, win = reversione su).

    Returns:
        dict con n, winrate, baseline, edge, mean, median, std, p25, p75,
        best, worst, tstat, pval.
    """
    n = int(len(fwd))
    res = dict(n=n, winrate=np.nan, baseline=baseline_win, edge=np.nan,
               mean=np.nan, median=np.nan, std=np.nan, p25=np.nan, p75=np.nan,
               best=np.nan, worst=np.nan, tstat=np.nan, pval=np.nan)
    if n == 0:
        return res

    wins = (fwd < 0).sum() if direction == "up" else (fwd > 0).sum()
    winrate = wins / n
    res.update(
        winrate=float(winrate),
        edge=float(winrate - baseline_win) if np.isfinite(baseline_win) else np.nan,
        mean=float(np.mean(fwd)),
        median=float(np.median(fwd)),
        p25=float(np.percentile(fwd, 25)),
        p75=float(np.percentile(fwd, 75)),
        best=float(np.max(fwd)),
        worst=float(np.min(fwd)),
    )
    if n > 1:
        std = float(np.std(fwd, ddof=1))
        res["std"] = std
        if std > 0:
            t = stats.ttest_1samp(fwd, 0.0)
            res["tstat"] = float(t.statistic)
            res["pval"] = float(t.pvalue)
    return res


def sensitivity_counts(df_valid: pd.DataFrame, fwd: int,
                       levels=SENSITIVITY_LEVELS) -> pd.DataFrame:
    """Conta gli eventi indipendenti a diverse soglie N (mini-tabella sensibilità).

    Riusa lo z-score già calcolato nel frame (indipendente da N), così la
    griglia mostra subito quanti eventi otterresti abbassando/alzando la soglia.
    """
    z = df_valid["z"]
    pos = df_valid["pos"].values
    rows = []
    for nlev in levels:
        up_pos = pos[(z >= nlev).values]
        dn_pos = pos[(z <= -nlev).values]
        rows.append({
            "N σ": nlev,
            "Eventi sopra": len(apply_cooldown(up_pos, fwd)),
            "Eventi sotto": len(apply_cooldown(dn_pos, fwd)),
        })
    return pd.DataFrame(rows)


# === Orchestrazione per orizzonte =======================================

def analyze_horizon(daily_price: pd.Series, name: str, n_std: float,
                    mode: str) -> dict:
    """Esegue l'intero studio per un singolo orizzonte temporale.

    Args:
        daily_price: serie prezzi giornaliera (adjusted close).
        name:        chiave in HORIZONS ('Giornaliero'/'Settimanale'/'Mensile').
        n_std:       soglia in deviazioni standard.
        mode:        'Statica' o 'Adattiva'.

    Returns:
        dict con tutto il necessario per il rendering (vedi chiavi sotto).
        Se i dati sono insufficienti restituisce {'ok': False, 'reason': ...}.
    """
    cfg = HORIZONS[name]
    price = resample_close(daily_price, cfg["rule"])

    if len(price) < cfg["fwd"] + 5:
        return dict(ok=False, name=name, reason="Storico insufficiente per questo orizzonte.")

    ret_full = period_returns(price)            # distribuzione completa (istogramma)
    hist_center = float(ret_full.median())
    hist_sigma = float(ret_full.std())

    df_full, df_valid = build_return_frame(price, n_std, mode, cfg)
    price_vals = df_full["price"].values
    k = cfg["fwd"]

    # --- Eventi de-clusterizzati e finestre forward ---
    up_pos = apply_cooldown(df_valid.loc[df_valid["up"], "pos"].values, k)
    dn_pos = apply_cooldown(df_valid.loc[df_valid["down"], "pos"].values, k)
    up_fwd = forward_returns(price_vals, up_pos, k)
    dn_fwd = forward_returns(price_vals, dn_pos, k)

    # --- Baseline incondizionata ---
    base = baseline_forward(price_vals, k)

    up_metrics = direction_metrics(up_fwd, base["p_down"], "up")
    dn_metrics = direction_metrics(dn_fwd, base["p_up"], "down")

    # --- Stato attuale (ultima barra disponibile) ---
    current = None
    if len(df_valid) > 0:
        last = df_valid.iloc[-1]
        if np.isfinite(last["z"]):
            current = dict(
                date=df_valid.index[-1],
                ret=float(last["ret"]),
                z=float(last["z"]),
                up=bool(last["up"]),
                down=bool(last["down"]),
            )

    return dict(
        ok=True,
        name=name,
        fwd_label=cfg["fwd_label"],
        rule=cfg["rule"],
        mode=mode,
        n_std=n_std,
        sigma_levels=cfg["sigma_levels"],
        ret_full=ret_full,
        hist_center=hist_center,
        hist_sigma=hist_sigma,
        n_periods=int(len(price)),
        first_date=price.index[0],
        last_date=price.index[-1],
        current=current,
        up=up_metrics,
        down=dn_metrics,
        up_fwd=up_fwd,
        down_fwd=dn_fwd,
        baseline=base,
        sensitivity=sensitivity_counts(df_valid, k),
    )
