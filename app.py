"""
app.py — Dashboard Kriterion Quant: Analisi degli Eccessi.

Studia, per ogni asset EODHD, cosa è storicamente accaduto ai rendimenti dopo
che un periodo ha chiuso oltre N deviazioni standard (un *eccesso*), su tre
orizzonti: giornaliero (5g dopo), settimanale (4w dopo), mensile (3m dopo).

Filosofia di design: leggibilità a colpo d'occhio.
    1. Stato attuale  -> dove si trova ORA l'asset rispetto alle bande.
    2. Quadro sintetico -> matrice color-coded con winrate/edge/affidabilità.
    3. Dettaglio per orizzonte (tab) -> istogramma + metriche + sensibilità.

Le metriche fragili non vengono mai nascoste: un semaforo segnala quando il
numero di eventi è troppo basso per fidarsi (🟢 ≥30 · 🟡 10-29 · 🔴 <10).
"""

import json

import numpy as np
import pandas as pd
import streamlit as st

from src.data_fetcher import fetch_eod, get_price_series
from src.calculations import HORIZONS, analyze_horizon, reliability, build_export
from src.charts import build_returns_cascade, build_forward_histogram, COLORS

# ===================================================
# CONFIGURAZIONE PAGINA
# ===================================================
st.set_page_config(
    page_title="Analisi Eccessi | Kriterion Quant",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ===================================================
# API KEY (da st.secrets; fallback manuale per test locale)
# ===================================================
EODHD_API_KEY = st.secrets.get("EODHD_API_KEY", "")

# ===================================================
# SIDEBAR — PARAMETRI
# ===================================================
with st.sidebar:
    st.title("⚙️ Parametri")
    st.divider()

    ticker = st.text_input(
        "Ticker (formato EODHD)",
        value="GSPC.INDX",
        help="Esempi: GSPC.INDX (S&P 500), AAPL.US, TSLA.US, "
             "BTC-USD.CC, EURUSD.FOREX, ENI.MI",
    ).strip()

    n_std = st.slider(
        "Soglia eccesso (deviazioni standard)",
        min_value=1.0, max_value=3.0, value=2.0, step=0.25,
        help="Un periodo è un 'eccesso' quando il suo rendimento supera questa "
             "soglia. Più alta = eventi più estremi ma più rari.",
    )

    mode = st.radio(
        "Modalità soglia",
        options=["Adattiva", "Statica"],
        index=0,
        help="Adattiva: z-score sulla volatilità recente (regime-aware, "
             "consigliata per storici lunghi). Statica: mediana ± Nσ "
             "sull'intero campione.",
    )

    with st.expander("🔧 Avanzate"):
        use_full_history = st.checkbox("Usa tutto lo storico disponibile",
                                       value=True)
        if use_full_history:
            start_date = None
        else:
            start_date = st.date_input(
                "Data inizio", value=pd.Timestamp("2010-01-01")
            ).strftime("%Y-%m-%d")

        if not EODHD_API_KEY:
            EODHD_API_KEY = st.text_input(
                "EODHD API Key", type="password",
                help="Non configurata nei secrets: inseriscila qui per il test.",
            )

    st.divider()
    st.caption("📡 Dati: EODHD — storico completo")
    st.caption("🔬 Win direzionale · cooldown = orizzonte · baseline incondizionata")

# ===================================================
# HEADER
# ===================================================
st.title("📊 Analisi degli Eccessi")
st.markdown(
    "Misura cosa è storicamente accaduto ai rendimenti **dopo un eccesso** "
    "(chiusura oltre N deviazioni standard) su tre orizzonti temporali. "
    "Utile per valutare se gli estremi vengono *riassorbiti* (mean reversion) "
    "o *estesi* (momentum), e con quale winrate."
)

if not EODHD_API_KEY:
    st.warning("⚠️ Nessuna API key EODHD trovata. Configurala nei *secrets* "
               "(`EODHD_API_KEY`) oppure inseriscila nella sezione **Avanzate** "
               "della sidebar.")
    st.stop()

if not ticker:
    st.info("Inserisci un ticker nella sidebar per iniziare.")
    st.stop()

# ===================================================
# FETCH DATI
# ===================================================
with st.spinner(f"⏳ Caricamento storico di {ticker}..."):
    try:
        kwargs = {"start": start_date} if start_date else {}
        raw = fetch_eod(ticker, EODHD_API_KEY, **kwargs)
    except Exception as e:  # noqa: BLE001 — vogliamo mostrare qualsiasi errore API
        st.error(f"❌ Errore nel caricamento dati per `{ticker}`: {e}\n\n"
                 "Verifica il ticker (formato EODHD, es. `AAPL.US`) e la chiave API.")
        st.stop()

if raw.empty:
    st.warning(f"⚠️ Nessun dato trovato per `{ticker}`. Controlla il formato del ticker.")
    st.stop()

price = get_price_series(raw)
if len(price) < 30:
    st.warning(f"⚠️ Storico troppo breve per `{ticker}` ({len(price)} osservazioni).")
    st.stop()

# ===================================================
# ANALISI (tutti gli orizzonti)
# ===================================================
results = {name: analyze_horizon(price, name, n_std, mode) for name in HORIZONS}

# ---- Riepilogo testata ----
years = (price.index[-1] - price.index[0]).days / 365.25
c1, c2, c3, c4 = st.columns(4)
c1.metric("Ultimo prezzo", f"{price.iloc[-1]:,.2f}")
c2.metric("Storico", f"{years:.1f} anni")
c3.metric("Osservazioni", f"{len(price):,}")
c4.metric("Dal", price.index[0].strftime("%d/%m/%Y"))

# --- Export JSON per report LLM ---
export_payload = json.dumps(
    build_export(ticker, n_std, mode, price, results),
    ensure_ascii=False, indent=2, allow_nan=False, default=str,
)
ec1, ec2 = st.columns([1, 3])
with ec1:
    st.download_button(
        "⬇️ Scarica dati analisi (JSON)",
        data=export_payload,
        file_name=f"analisi_eccessi_{ticker.replace('.', '_')}_"
                  f"{pd.Timestamp.today():%Y%m%d}.json",
        mime="application/json",
        use_container_width=True,
        help="Tutte le metriche dello studio in JSON, pronte da dare a un LLM "
             "per generare un report.",
    )
with ec2:
    st.caption("Include un **prompt pronto** (`prompt_report`) + metriche, winrate, "
               "edge, baseline, stato attuale, sensibilità e statistiche "
               "distribuzionali per tutti gli orizzonti, con legenda dei campi. "
               "Consegnalo a un LLM così com'è per ottenere il report.")

st.divider()


# ===================================================
# 1) STATO ATTUALE — dove si trova ORA l'asset
# ===================================================
def state_label(cur: dict | None, n_std: float) -> tuple[str, str]:
    """Etichetta e colore dello stato attuale rispetto alle soglie."""
    if cur is None:
        return "n/d", COLORS["neutral"]
    if cur["up"]:
        return f"ECCESSO SOPRA (z={cur['z']:+.2f})", COLORS["negative"]
    if cur["down"]:
        return f"ECCESSO SOTTO (z={cur['z']:+.2f})", COLORS["positive"]
    return f"Normale (z={cur['z']:+.2f})", COLORS["neutral"]


st.subheader("📍 Stato attuale")
st.caption("Posizione dell'ultimo periodo rispetto alle bande di deviazione standard. "
           "Un eccesso *in corso* è il punto di partenza operativo dello studio.")
cols = st.columns(3)
for col, name in zip(cols, HORIZONS):
    res = results[name]
    cur = res.get("current") if res.get("ok") else None
    label, color = state_label(cur, n_std)
    last_ret = f"{cur['ret']*100:+.2f}%" if cur else "—"
    col.metric(f"{name}", last_ret)
    col.markdown(
        f"<div style='background:{color}22;border-left:4px solid {color};"
        f"padding:6px 10px;border-radius:4px;font-size:13px;color:{COLORS['text']}'>"
        f"{label}</div>", unsafe_allow_html=True,
    )

st.divider()


# ===================================================
# 2) RENDIMENTI STORICI — cascata a barre (vista d'insieme)
# ===================================================
st.subheader("📊 Rendimenti storici — cascata giornaliero · settimanale · mensile")
st.caption(
    "Ogni **barra** è il rendimento di un periodo lungo tutto lo storico. "
    "Le linee di **mediana** (bianca) e **±Nσ** (arancio) sono quelle usate per la "
    "rilevazione: **piatte** in modalità Statica, **ondulate** in Adattiva (si "
    "ricalcolano ad ogni barra sulla volatilità recente). La linea **gialla** è il "
    "**valore attuale**. Barre oltre soglia evidenziate (🔴 eccesso sopra · 🟢 sotto). "
    "I tre grafici condividono l'asse temporale per confrontare gli orizzonti a colpo d'occhio."
)
st.plotly_chart(build_returns_cascade(results, list(HORIZONS)),
                use_container_width=True)

st.divider()


# ===================================================
# 3) QUADRO SINTETICO — matrice color-coded
# ===================================================
st.subheader("🧭 Quadro sintetico")
st.caption("Il numero che conta è l'**edge** (winrate condizionato − baseline). "
           "Il semaforo indica se il campione è abbastanza ampio per fidarsi.")


def build_summary(direction: str) -> pd.DataFrame:
    """Costruisce la tabella riassuntiva per una direzione (up/down)."""
    rows = []
    for name in HORIZONS:
        res = results[name]
        if not res.get("ok"):
            continue
        m = res[direction]
        emoji, _ = reliability(m["n"])
        rows.append({
            "Orizzonte": name,
            "Affid.": emoji,
            "N eventi": m["n"],
            "Winrate": m["winrate"],
            "Baseline": m["baseline"],
            "Edge": m["edge"],
            "Mediana fwd": m["median"],
            "p-value": m["pval"],
        })
    cols = ["Orizzonte", "Affid.", "N eventi", "Winrate", "Baseline",
            "Edge", "Mediana fwd", "p-value"]
    if not rows:
        return pd.DataFrame(columns=cols).set_index("Orizzonte")
    return pd.DataFrame(rows).set_index("Orizzonte")


def style_summary(df: pd.DataFrame):
    """Formattazione + color coding della tabella sintetica."""
    fmt = {
        "Winrate": "{:.1%}", "Baseline": "{:.1%}", "Edge": "{:+.1%}",
        "Mediana fwd": "{:+.2%}", "p-value": "{:.3f}", "N eventi": "{:.0f}",
    }

    def edge_color(v):
        if pd.isna(v):
            return ""
        return f"color: {COLORS['positive']}" if v > 0 else f"color: {COLORS['negative']}"

    return (
        df.style.format(fmt, na_rep="—")
        .map(edge_color, subset=["Edge"])
        .background_gradient(cmap="RdYlGn", subset=["Winrate"], vmin=0.3, vmax=0.7)
        .set_properties(**{"text-align": "center", "font-size": "13px"})
    )


sc1, sc2 = st.columns(2)
with sc1:
    st.markdown(f"##### 🔴 Eccesso **SOPRA** &nbsp;·&nbsp; win = reversione ↓")
    st.dataframe(style_summary(build_summary("up")), use_container_width=True)
with sc2:
    st.markdown(f"##### 🟢 Eccesso **SOTTO** &nbsp;·&nbsp; win = reversione ↑")
    st.dataframe(style_summary(build_summary("down")), use_container_width=True)

st.divider()


# ===================================================
# 4) DETTAGLIO PER ORIZZONTE (tab) — lo studio forward
# ===================================================
st.subheader("🔬 Cosa è successo dopo l'eccesso")


def render_direction(res: dict, direction: str):
    """Blocco dettagliato di una direzione: metriche + istogramma forward."""
    m = res[direction]
    emoji, rel_label = reliability(m["n"])
    arrow = "↓" if direction == "up" else "↑"
    title = "Eccesso SOPRA" if direction == "up" else "Eccesso SOTTO"

    st.markdown(f"**{title}** &nbsp; {emoji} *{rel_label}* — {m['n']} eventi indipendenti")

    if m["n"] == 0:
        st.info("Nessun eccesso rilevato a questa soglia. Prova ad abbassare N "
                "(vedi tabella di sensibilità).")
        return

    k1, k2, k3 = st.columns(3)
    k1.metric(f"Winrate reversione {arrow}",
              f"{m['winrate']*100:.1f}%",
              delta=f"{m['edge']*100:+.1f} pp vs baseline"
              if np.isfinite(m["edge"]) else None)
    k2.metric("Mediana forward", f"{m['median']*100:+.2f}%")
    pval_txt = f"{m['pval']:.3f}" if np.isfinite(m["pval"]) else "—"
    k3.metric("p-value (media≠0)", pval_txt)

    fwd = res["up_fwd"] if direction == "up" else res["down_fwd"]
    st.plotly_chart(
        build_forward_histogram(fwd, direction, res["fwd_label"], m["winrate"]),
        use_container_width=True,
    )


tabs = st.tabs([f"{name} — {HORIZONS[name]['fwd_label']} dopo" for name in HORIZONS])
for tab, name in zip(tabs, HORIZONS):
    res = results[name]
    with tab:
        if not res.get("ok"):
            st.warning(f"⚠️ {res.get('reason', 'Dati insufficienti per questo orizzonte.')}")
            continue

        st.caption(
            f"Rilevazione eventi in modalità **{res['mode']}** · "
            f"soglia **{res['n_std']:g}σ** · "
            f"forward **{res['fwd_label']}** · "
            f"cooldown = {HORIZONS[name]['fwd']} periodi (finestre indipendenti)."
        )

        d1, d2 = st.columns(2)
        with d1:
            render_direction(res, "up")
        with d2:
            render_direction(res, "down")

        st.markdown("###### 📐 Sensibilità alla soglia N")
        st.caption("Quanti eventi indipendenti otterresti a diverse soglie. "
                   "Utile sugli asset con poco storico per scegliere una N che dia "
                   "abbastanza eventi senza snaturare il concetto di 'eccesso'.")
        sens = res["sensitivity"].copy()
        st.dataframe(
            sens.style.format({"N σ": "{:.1f}", "Eventi sopra": "{:.0f}",
                               "Eventi sotto": "{:.0f}"}),
            use_container_width=True, hide_index=True,
        )

# ===================================================
# METODOLOGIA
# ===================================================
with st.expander("ℹ️ Metodologia e note tecniche"):
    st.markdown("""
**Dati.** Prezzi *adjusted close* da EODHD (dividendi e split inclusi), tutto lo
storico disponibile. Settimanale = chiusura venerdì; mensile = fine mese.

**Definizione di eccesso.**
- *Statica*: il rendimento di periodo supera `mediana ± N·σ` calcolati
  sull'intero campione. Coincide con le bande mostrate sull'istogramma.
- *Adattiva* (default): il rendimento, standardizzato sulla volatilità recente
  (mediana e σ su finestra mobile, calcolate solo su dati passati), supera N in
  z-score. Regime-aware: confronta epoche con volatilità diversa in modo equo.
  Su storici corti la finestra si comporta come *expanding* per non perdere dati.

**Win direzionale.** Dopo un eccesso SOPRA il win è un rendimento forward
negativo (reversione ↓); dopo un eccesso SOTTO, positivo (reversione ↑).

**Indipendenza dei campioni.** Dopo ogni evento si applica un *cooldown* pari
all'orizzonte forward (5g / 4w / 3m): le finestre forward non si sovrappongono,
evitando autocorrelazione che gonfierebbe la significatività.

**Edge.** Ogni winrate è confrontato con la **baseline incondizionata** (la
probabilità di win entrando in un periodo qualsiasi). `Edge = winrate − baseline`
è la misura del vantaggio reale.

**Affidabilità.** 🟢 ≥ 30 eventi · 🟡 10–29 · 🔴 < 10. Sotto i 10 eventi i numeri
sono mostrati ma vanno trattati come aneddotici.

**Avvertenze.** Studio descrittivo su dati storici, non un sistema di trading.
Volatility clustering, cambi di regime e costi di transazione non sono modellati.
""")

st.caption("Kriterion Quant · Analisi degli Eccessi · dati EODHD")
