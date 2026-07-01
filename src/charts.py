"""
charts.py — Grafici Plotly standardizzati (tema scuro Kriterion Quant).

Due grafici principali:
    1. build_returns_cascade   -> tre grafici a barre impilati (giornaliero /
       settimanale / mensile): ogni barra è il rendimento di un periodo, con
       linee orizzontali di mediana, ±Nσ e valore attuale. È la vista d'insieme
       per capire a colpo d'occhio dove siamo rispetto allo storico.
    2. build_forward_histogram -> distribuzione dei rendimenti forward dopo un
       eccesso, per smascherare reversione vs momentum.

Tutte le funzioni ricevono dati già calcolati e restituiscono go.Figure.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Colore delle barre "normali" (dentro le bande): blu-grigio tenue
BAR_NORMAL = "rgba(120,140,190,0.55)"

COLORS = {
    "primary":    "#2196F3",
    "secondary":  "#FF9800",
    "positive":   "#4CAF50",
    "negative":   "#F44336",
    "neutral":    "#9E9E9E",
    "background": "#1E1E2E",
    "surface":    "#2A2A3E",
    "text":       "#E0E0E0",
    "accent":     "#FFEB3B",   # giallo — posizione attuale (massima evidenza)
}


def _base_layout(title: str, x_title: str = "", y_title: str = "",
                 height: int = 360) -> dict:
    """Layout Plotly condiviso, ottimizzato per lettura immediata."""
    return dict(
        title=dict(text=title, font=dict(size=15, color=COLORS["text"])),
        paper_bgcolor=COLORS["background"],
        plot_bgcolor=COLORS["surface"],
        font=dict(color=COLORS["text"], family="Inter, Arial, sans-serif"),
        xaxis=dict(title=x_title, showgrid=True, gridcolor="#333355",
                   zeroline=False, color=COLORS["text"]),
        yaxis=dict(title=y_title, showgrid=True, gridcolor="#333355",
                   zeroline=False, color=COLORS["text"]),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#444466",
                    orientation="h", yanchor="bottom", y=1.02, x=0),
        margin=dict(l=55, r=20, t=70, b=50),
        height=height,
        bargap=0.02,
    )


def build_returns_cascade(results: dict, order: list[str]) -> go.Figure:
    """Cascata di 3 grafici a barre dei rendimenti nel tempo (D / W / M).

    Ogni barra è il rendimento di un periodo. Le tre righe condividono l'asse
    temporale, così eccessi allineati nel tempo sui diversi orizzonti si leggono
    a colpo d'occhio.

    Le bande σ sono quelle EFFETTIVAMENTE usate per la rilevazione (mode-aware):
        - mediana (bianca) e ±Nσ (arancio) disegnate come linee nel tempo:
          piatte in modalità Statica, ondulate in modalità Adattiva (si
          ricalcolano ad ogni barra sulla volatilità recente);
        - altri livelli σ di contesto (grigio tratteggiato);
        - valore attuale (linea gialla orizzontale) → dove siamo ORA rispetto a
          tutto lo storico.
    Le barre oltre la soglia per-barra sono evidenziate: rosso = eccesso sopra,
    verde = eccesso sotto. Una barra colorata sta dunque esattamente oltre la
    linea ±Nσ disegnata nello stesso istante.

    Args:
        results: dict {nome_orizzonte: output di analyze_horizon()}.
        order:   ordine degli orizzonti dall'alto verso il basso.
    """
    fig = make_subplots(rows=len(order), cols=1, shared_xaxes=True,
                        vertical_spacing=0.06)

    for i, name in enumerate(order, start=1):
        res = results[name]
        first_row = (i == 1)  # legenda mostrata solo sulla prima riga

        # Etichetta orizzonte dentro il pannello (in alto a sinistra)
        cur = res.get("current") if res.get("ok") else None
        lbl = name
        if cur is not None:
            lbl += f" · ora {cur['ret']*100:+.2f}% (z={cur['z']:+.2f})"
        fig.add_annotation(text=f"<b>{lbl}</b>", row=i, col=1,
                           xref="x domain", yref="y domain", x=0.01, y=0.97,
                           showarrow=False, xanchor="left",
                           font=dict(size=12, color=COLORS["text"]))

        if not res.get("ok"):
            continue

        ret = res["ret_full"] * 100.0
        idx = ret.index
        n_std = res["n_std"]
        bc = res["band_center"] * 100.0          # centro per-barra (Series)
        bs = res["band_sigma"] * 100.0           # sigma per-barra (Series)
        upper = bc + n_std * bs
        lower = bc - n_std * bs

        # --- Barre colorate per eccesso (soglia per-barra) ---
        vals = ret.values
        up_v = np.where(np.isnan(upper.values), np.inf, upper.values)
        dn_v = np.where(np.isnan(lower.values), -np.inf, lower.values)
        colors = np.where(vals >= up_v, COLORS["negative"],
                          np.where(vals <= dn_v, COLORS["positive"], BAR_NORMAL))
        fig.add_trace(go.Bar(
            x=idx, y=vals, marker_color=colors, marker_line_width=0,
            showlegend=False,
            hovertemplate="%{x|%d/%m/%Y}<br>Rendimento: %{y:.2f}%<extra></extra>",
        ), row=i, col=1)

        # --- Bande σ di contesto (livelli diversi dalla soglia N) ---
        for k in res["sigma_levels"]:
            if abs(k - n_std) < 1e-9:
                continue
            for band in (bc + k * bs, bc - k * bs):
                fig.add_trace(go.Scatter(
                    x=idx, y=band.values, mode="lines", showlegend=False,
                    line=dict(color=COLORS["neutral"], width=0.8, dash="dot"),
                    opacity=0.4, hoverinfo="skip",
                ), row=i, col=1)

        # --- Mediana / centro (linea nel tempo) ---
        fig.add_trace(go.Scatter(
            x=idx, y=bc.values, mode="lines", name="mediana",
            legendgroup="med", showlegend=first_row, hoverinfo="skip",
            line=dict(color=COLORS["text"], width=1.3),
        ), row=i, col=1)

        # --- ±Nσ (soglia selezionata, evidenziata) ---
        fig.add_trace(go.Scatter(
            x=idx, y=upper.values, mode="lines", name=f"±{n_std:g}σ",
            legendgroup="nsig", showlegend=first_row, hoverinfo="skip",
            line=dict(color=COLORS["secondary"], width=1.4),
        ), row=i, col=1)
        fig.add_trace(go.Scatter(
            x=idx, y=lower.values, mode="lines", name=f"-{n_std:g}σ",
            legendgroup="nsig", showlegend=False, hoverinfo="skip",
            line=dict(color=COLORS["secondary"], width=1.4),
        ), row=i, col=1)

        # --- Valore attuale (linea orizzontale gialla) ---
        if cur is not None:
            fig.add_hline(y=cur["ret"] * 100.0, row=i, col=1,
                          line=dict(color=COLORS["accent"], width=2),
                          annotation_text="ORA", annotation_position="top right",
                          annotation_font=dict(size=10, color=COLORS["accent"]))

        fig.update_yaxes(title_text="Rend. %", row=i, col=1,
                         gridcolor="#333355", color=COLORS["text"], zeroline=False)

    fig.update_xaxes(gridcolor="#333355", color=COLORS["text"])
    fig.update_layout(
        title=dict(text="Rendimenti storici per periodo — bande σ (ricalcolate) e posizione attuale",
                   font=dict(size=16, color=COLORS["text"])),
        paper_bgcolor=COLORS["background"], plot_bgcolor=COLORS["surface"],
        font=dict(color=COLORS["text"], family="Inter, Arial, sans-serif"),
        legend=dict(orientation="h", yanchor="bottom", y=1.005, x=0,
                    bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=60, r=40, t=80, b=40),
        height=300 * len(order), bargap=0.0,
    )
    return fig


def build_forward_histogram(fwd: np.ndarray, direction: str,
                            fwd_label: str, winrate: float | None) -> go.Figure:
    """Distribuzione dei rendimenti forward dopo un eccesso.

    Rende visibile se l'eccesso si riassorbe (massa dalla parte della reversione)
    o continua (massa nella stessa direzione dell'eccesso → momentum).

    Args:
        fwd:       array dei rendimenti forward degli eventi.
        direction: 'up' (eccesso sopra) o 'down' (eccesso sotto).
        fwd_label: etichetta orizzonte forward (es. '5 giorni').
        winrate:   winrate di reversione, per l'annotazione (può essere None).
    """
    title = ("Dopo eccesso SOPRA → rendimenti a "
             if direction == "up" else "Dopo eccesso SOTTO → rendimenti a ") + fwd_label

    fig = go.Figure()
    if fwd is None or len(fwd) == 0:
        fig.update_layout(**_base_layout(title, height=300))
        fig.add_annotation(text="Nessun evento", showarrow=False,
                           font=dict(color=COLORS["neutral"], size=13))
        return fig

    vals = fwd * 100.0
    fig.add_trace(go.Histogram(
        x=vals, nbinsx=30, marker_color=COLORS["primary"], opacity=0.75,
        hovertemplate="Forward: %{x:.2f}%<extra></extra>",
    ))

    # Linea dello zero (confine win/lose) e della media
    fig.add_vline(x=0, line=dict(color=COLORS["neutral"], width=1.5, dash="dash"))
    fig.add_vline(x=float(vals.mean()), line=dict(color=COLORS["accent"], width=2),
                  annotation_text=f"media {vals.mean():+.2f}%",
                  annotation_position="top",
                  annotation_font=dict(size=10, color=COLORS["accent"]))

    if winrate is not None and np.isfinite(winrate):
        arrow = "↓" if direction == "up" else "↑"
        fig.add_annotation(
            x=0.02, y=0.95, xref="paper", yref="paper", showarrow=False,
            align="left", text=f"Winrate reversione {arrow}: <b>{winrate*100:.1f}%</b>",
            bgcolor="rgba(42,42,62,0.9)", bordercolor="#444466",
            font=dict(size=12, color=COLORS["text"]),
        )

    fig.update_layout(**_base_layout(
        title, x_title=f"Rendimento forward a {fwd_label} (%)",
        y_title="Frequenza", height=300))
    fig.update_layout(showlegend=False)
    return fig
