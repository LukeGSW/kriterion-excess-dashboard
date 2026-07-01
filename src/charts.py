"""
charts.py — Grafici Plotly standardizzati (tema scuro Kriterion Quant).

Due grafici principali:
    1. build_excess_histogram  -> distribuzione dei rendimenti di periodo con
       mediana, bande ±Nσ, zone d'eccesso ombreggiate e posizione attuale.
    2. build_forward_histogram -> distribuzione dei rendimenti forward dopo un
       eccesso, per smascherare reversione vs momentum.

Tutte le funzioni ricevono dati già calcolati e restituiscono go.Figure.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

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


def build_excess_histogram(res: dict) -> go.Figure:
    """Istogramma dei rendimenti di periodo con bande σ e posizione attuale.

    Le bande (mediana ± kσ) usano sempre le statistiche full-sample: sono il
    riferimento visivo della distribuzione, indipendentemente dalla modalità di
    rilevazione scelta. La banda alla soglia N selezionata è evidenziata; le
    altre sono tratteggiate leggere per contesto.

    Args:
        res: output di calculations.analyze_horizon().
    """
    ret = res["ret_full"] * 100.0           # in percentuale
    center = res["hist_center"] * 100.0
    sigma = res["hist_sigma"] * 100.0
    n_std = res["n_std"]
    levels = res["sigma_levels"]

    upper_n = center + n_std * sigma
    lower_n = center - n_std * sigma

    fig = go.Figure()

    # Zone d'eccesso ombreggiate (rosso = ipercomprato sopra, verde = ipervenduto sotto)
    x_max = float(ret.max())
    x_min = float(ret.min())
    fig.add_vrect(x0=upper_n, x1=x_max, fillcolor=COLORS["negative"],
                  opacity=0.10, line_width=0, layer="below")
    fig.add_vrect(x0=x_min, x1=lower_n, fillcolor=COLORS["positive"],
                  opacity=0.10, line_width=0, layer="below")

    # Istogramma dei rendimenti
    fig.add_trace(go.Histogram(
        x=ret, nbinsx=80, name="Rendimenti",
        marker_color=COLORS["primary"], opacity=0.75,
        hovertemplate="Rendimento: %{x:.2f}%<br>Frequenza: %{y}<extra></extra>",
    ))

    # Bande σ di contesto (tratteggiate leggere), escludendo la soglia N
    for k in levels:
        if abs(k - n_std) < 1e-9:
            continue
        for sign in (+1, -1):
            xpos = center + sign * k * sigma
            lab = f"{'+' if sign > 0 else '-'}{k}σ"
            fig.add_vline(x=xpos, line=dict(color=COLORS["neutral"], width=1,
                          dash="dot"), opacity=0.45,
                          annotation_text=lab, annotation_position="top",
                          annotation_font=dict(size=9, color=COLORS["neutral"]))

    # Mediana
    fig.add_vline(x=center, line=dict(color=COLORS["text"], width=1.5),
                  annotation_text="Mediana", annotation_position="top",
                  annotation_font=dict(size=10, color=COLORS["text"]))

    # Soglie N selezionate (evidenziate)
    for xpos, lab in [(upper_n, f"+{n_std:g}σ"), (lower_n, f"-{n_std:g}σ")]:
        fig.add_vline(x=xpos, line=dict(color=COLORS["secondary"], width=2),
                      annotation_text=lab, annotation_position="top",
                      annotation_font=dict(size=11, color=COLORS["secondary"]))

    # Posizione attuale (massima evidenza)
    cur = res.get("current")
    if cur is not None:
        xpos = cur["ret"] * 100.0
        zlab = f"Ora: {xpos:+.2f}%  (z={cur['z']:+.2f})"
        fig.add_vline(x=xpos, line=dict(color=COLORS["accent"], width=2.5),
                      annotation_text=zlab, annotation_position="bottom",
                      annotation_font=dict(size=11, color=COLORS["accent"]))

    fig.update_layout(**_base_layout(
        f"Distribuzione rendimenti {res['name'].lower()} — bande σ full-sample",
        x_title="Rendimento di periodo (%)", y_title="Frequenza", height=380,
    ))
    fig.update_layout(showlegend=False)
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
