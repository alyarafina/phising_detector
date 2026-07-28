"""
app.py
-------
Linkora — premium SaaS-style front-end for the phishing URL detection engine.

IMPORTANT: This file only changes presentation (layout, styling, copy, UX flow).
All machine-learning logic — feature extraction, scaling, model loading and
prediction — is untouched and delegated to feature_extractor.py and
model_utils.py exactly as before.
"""

import time

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from feature_extractor import extract_features
from model_utils import load_artifacts, load_all_models, predict_all

# --------------------------------------------------------------------------------
# Page config
# --------------------------------------------------------------------------------
st.set_page_config(
    page_title="Linkora — Phishing URL Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --------------------------------------------------------------------------------
# Presentation-only helpers (do NOT affect the underlying prediction)
# --------------------------------------------------------------------------------
def risk_profile(prob_phishing: float, label: int):
    """Maps an existing model score to a display-only risk tier + recommendation.
    Purely cosmetic — uses only values already produced by predict_all()."""
    if label == 0 and prob_phishing < 15:
        return "Low", "This URL shows very few phishing indicators. It appears safe to visit, but always verify the domain before entering credentials."
    if label == 0:
        return "Guarded", "This URL leans legitimate, though a few signals are borderline. Proceed normally, but stay alert for anything unusual on the page."
    if label == 1 and prob_phishing < 75:
        return "Elevated", "This URL shows several phishing indicators. Avoid entering passwords or personal data until you can verify it independently."
    return "Critical", "This URL strongly matches known phishing patterns. Do not enter credentials, payment details, or personal information."


# --------------------------------------------------------------------------------
# Global styling
# --------------------------------------------------------------------------------
st.markdown(
    """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">

<style>
:root{
    --ink:#3d2f22;
    --ink-soft:#6b5a48;
    --ink-faint:#a4917d;
    --surface:#ffffff;
    --surface-alt:#faf6ef;
    --border:#e8ddcd;
    --accent:#8a6d46;
    --accent-soft:#f3ead9;
    --safe:#3f8556;
    --safe-soft:#eaf6ee;
    --danger:#b0503b;
    --danger-soft:#fbeee9;
    --radius:16px;
    --shadow:0 1px 2px rgba(90,70,40,.05), 0 8px 24px rgba(90,70,40,.07);
}

html, body, [class*="css"]{
    font-family:'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    color:var(--ink);
}

.stApp{ background:var(--surface-alt); }
#MainMenu, footer, header[data-testid="stHeader"]{visibility:hidden; height:0;}
section[data-testid="stSidebar"]{ display:none; }
.block-container{padding-top:1rem; max-width:1080px;}

/* ---------------- Responsive: tablet & mobile ---------------- */
@media (max-width: 900px){
    .block-container{ padding-left:1rem !important; padding-right:1rem !important; }
    div[data-testid="stHorizontalBlock"]{ flex-wrap:wrap !important; }
    div[data-testid="stHorizontalBlock"] > div[data-testid="column"]{
        min-width:100% !important; flex:1 1 100% !important;
    }
}
@media (max-width: 640px){
    .linkora-nav{ flex-wrap:wrap; padding:12px 16px; gap:8px; }
    .linkora-nav-links{ display:none; }
    .hero-title{ font-size:1.9rem !important; }
    .hero-sub{ font-size:.95rem !important; padding:0 6px; }
    .hero-wrap{ padding:24px 4px 4px 4px; }
    .scan-card{ padding:18px 16px; }
    .section-card{ padding:16px 14px; }
    .step-card{ padding:18px; }
    .model-card{ padding:16px; }
    .ring{ width:96px; height:96px; }
    .ring-inner{ width:76px; height:76px; }
    .verdict-banner{ padding:18px 16px; flex-direction:column; align-items:flex-start; }
}

h1,h2,h3,h4,h5{
    font-family:'Manrope', 'Inter', sans-serif;
    letter-spacing:-0.02em;
    color:var(--ink);
}

/* ---------------- Navbar ---------------- */
.linkora-nav{
    position:sticky; top:0; z-index:999;
    display:flex; align-items:center; justify-content:space-between;
    padding:14px 28px;
    background:rgba(255,255,255,.85);
    backdrop-filter:blur(14px);
    border:1px solid var(--border);
    border-radius:14px;
    margin:8px 0 28px 0;
    box-shadow:var(--shadow);
}
.linkora-logo{
    display:flex; align-items:center; gap:10px;
    font-family:'Manrope', sans-serif;
    font-weight:800; font-size:1.15rem; color:var(--ink);
}
.linkora-logo .mark{
    width:30px; height:30px; border-radius:9px;
    background:var(--ink);
    display:flex; align-items:center; justify-content:center;
    color:#fff; font-size:15px;
}
.linkora-nav-links{ display:flex; gap:26px; font-size:.9rem; font-weight:600; color:var(--ink-soft); }
.linkora-badge{
    font-size:.72rem; font-weight:700; padding:4px 10px; border-radius:999px;
    background:var(--accent-soft); color:var(--ink-soft);
    border:1px solid var(--border);
}

/* ---------------- Hero ---------------- */
.hero-wrap{ text-align:center; padding:36px 12px 12px 12px; animation:fadeSlideUp .6s ease; }
.hero-eyebrow{
    display:inline-flex; align-items:center; gap:6px;
    font-size:.78rem; font-weight:700; color:var(--ink-soft);
    background:var(--accent-soft); padding:6px 14px; border-radius:999px;
    margin-bottom:18px; border:1px solid var(--border);
}
.hero-title{
    font-size:2.9rem; font-weight:800; line-height:1.12; margin:0 0 14px 0;
    letter-spacing:-0.03em;
}
.hero-sub{
    max-width:640px; margin:0 auto; color:var(--ink-soft); font-size:1.05rem; line-height:1.6;
}
@keyframes fadeSlideUp{
    from{opacity:0; transform:translateY(14px);}
    to{opacity:1; transform:translateY(0);}
}
@keyframes fadeIn{ from{opacity:0;} to{opacity:1;} }
@keyframes pulseDot{ 0%,100%{opacity:.35;} 50%{opacity:1;} }

/* ---------------- Scan card / input ---------------- */
.scan-card{
    background:var(--surface); border:1px solid var(--border); border-radius:var(--radius);
    padding:26px 28px; box-shadow:var(--shadow); margin:30px 0 8px 0;
    animation:fadeSlideUp .7s ease;
}
.scan-card-label{
    font-weight:700; font-size:1rem; margin-bottom:10px; display:flex; align-items:center; gap:8px;
}
div[data-testid="stTextInput"] input{
    border-radius:12px !important; border:1.5px solid var(--border) !important;
    padding:14px 16px !important; font-size:1rem !important;
    background:var(--surface-alt) !important;
    transition:border-color .15s ease, box-shadow .15s ease;
}
div[data-testid="stTextInput"] input:focus{
    border-color:var(--ink) !important;
    box-shadow:0 0 0 3px var(--accent-soft) !important;
}
.stButton>button, .stButton>button:focus{
    background:var(--ink) !important; color:#fff !important;
    border-radius:12px !important; border:none !important;
    padding:14px 20px !important; font-weight:700 !important; font-size:.95rem !important;
    transition:transform .12s ease, opacity .12s ease;
    box-shadow:0 4px 14px rgba(11,13,16,.18) !important;
}
.stButton>button:hover{ transform:translateY(-1px); opacity:.9; }

/* ---------------- Settings row (replaces sidebar) ---------------- */
.settings-label{ font-size:.82rem; font-weight:700; color:var(--ink-soft); margin:2px 0 6px 2px; }
div[data-testid="stExpander"]{
    border:1px solid var(--border) !important; border-radius:12px !important;
    background:var(--surface) !important; box-shadow:var(--shadow);
}

/* ---------------- How it works ---------------- */
.section-heading{ text-align:center; margin:56px 0 26px 0; }
.section-heading h2{ font-size:1.7rem; margin-bottom:6px; }
.section-heading p{ color:var(--ink-soft); }
.step-card{
    background:var(--surface); border:1px solid var(--border); border-radius:var(--radius);
    padding:24px; height:100%; box-shadow:var(--shadow);
    transition:transform .18s ease, box-shadow .18s ease;
}
.step-card:hover{ transform:translateY(-4px); box-shadow:0 12px 30px rgba(16,17,20,.09); }
.step-num{
    width:36px; height:36px; border-radius:10px; background:var(--accent-soft); color:var(--ink);
    display:flex; align-items:center; justify-content:center; font-weight:800; margin-bottom:14px;
    font-size:.95rem;
}
.step-card h4{ font-size:1.02rem; margin:0 0 6px 0; }
.step-card p{ font-size:.87rem; color:var(--ink-soft); line-height:1.55; margin:0; }

/* ---------------- Scan progress (loading experience) ---------------- */
.scan-progress{ animation:fadeIn .3s ease; }
.scan-progress-title{
    font-weight:700; font-size:1rem; display:flex; align-items:center; gap:10px; margin-bottom:16px;
}
.scan-spinner{
    width:16px; height:16px; border-radius:50%; border:2px solid var(--border);
    border-top-color:var(--ink); animation:spin .8s linear infinite;
}
@keyframes spin{ to{ transform:rotate(360deg); } }
.scan-step{
    display:flex; align-items:center; gap:12px; padding:10px 4px; font-size:.9rem; color:var(--ink-faint);
    transition:color .2s ease;
}
.scan-step.active{ color:var(--ink); font-weight:600; }
.scan-step.done{ color:var(--ink-soft); font-weight:600; }
.scan-dot{ width:9px; height:9px; border-radius:50%; background:var(--border); flex:none; transition:background .2s ease; }
.scan-step.active .scan-dot{ background:var(--ink); animation:pulseDot 1s ease infinite; }
.scan-step.done .scan-dot{ background:var(--safe); }
.scan-bar-track{ height:4px; border-radius:999px; background:var(--border); margin-top:6px; overflow:hidden; }
.scan-bar-fill{ height:100%; background:var(--ink); border-radius:999px; transition:width .3s ease; }

/* ---------------- Verdict banner ---------------- */
.verdict-banner{
    border-radius:var(--radius); padding:22px 28px; margin:26px 0 18px 0;
    display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:12px;
    animation:fadeSlideUp .5s ease; box-shadow:var(--shadow);
}
.verdict-safe{ background:var(--safe-soft); border:1px solid rgba(22,163,74,.2); }
.verdict-danger{ background:var(--danger-soft); border:1px solid rgba(220,38,38,.2); }
.verdict-title{ font-size:1.3rem; font-weight:800; margin:0; }
.verdict-safe .verdict-title{ color:var(--safe); }
.verdict-danger .verdict-title{ color:var(--danger); }
.verdict-sub{ font-size:.85rem; color:var(--ink-soft); margin-top:2px; }

/* ---------------- Model result cards ---------------- */
.model-card{
    background:var(--surface); border:1px solid var(--border); border-radius:var(--radius);
    padding:22px; box-shadow:var(--shadow); text-align:center; animation:fadeSlideUp .55s ease;
    transition:transform .18s ease, box-shadow .18s ease;
}
.model-card:hover{ transform:translateY(-3px); box-shadow:0 12px 30px rgba(16,17,20,.09); }
.model-name{ font-weight:700; font-size:.95rem; margin-bottom:10px; color:var(--ink-soft); }
.ring{
    width:112px; height:112px; border-radius:50%; margin:0 auto 12px auto;
    display:flex; align-items:center; justify-content:center; position:relative;
}
.ring-inner{
    width:88px; height:88px; border-radius:50%; background:var(--surface);
    display:flex; flex-direction:column; align-items:center; justify-content:center;
}
.ring-pct{ font-size:1.2rem; font-weight:800; }
.ring-caption{ font-size:.65rem; color:var(--ink-soft); font-weight:600; }
.pill{
    display:inline-block; padding:5px 14px; border-radius:999px; font-size:.78rem; font-weight:700;
    margin-top:4px;
}
.pill-safe{ background:var(--safe-soft); color:var(--safe); }
.pill-danger{ background:var(--danger-soft); color:var(--danger); }
.model-conf{ font-size:.78rem; color:var(--ink-soft); margin-top:8px; }

/* ---------------- Detail chips (Risk Level + Recommendation) ---------------- */
.card-divider{ height:1px; background:var(--border); margin:16px 0 14px 0; }
.detail-row{ display:flex; align-items:center; justify-content:space-between; font-size:.78rem; margin-bottom:8px; }
.detail-key{ color:var(--ink-faint); font-weight:600; }
.detail-val{ font-weight:700; }
.risk-chip{ padding:3px 10px; border-radius:999px; font-size:.72rem; font-weight:700; }
.risk-Low, .risk-Guarded{ background:var(--safe-soft); color:var(--safe); }
.risk-Elevated, .risk-Critical{ background:var(--danger-soft); color:var(--danger); }
.model-reco{ font-size:.76rem; color:var(--ink-soft); line-height:1.5; text-align:left; margin-top:4px; }

/* ---------------- Misc containers ---------------- */
div[data-testid="stDataFrame"]{
    border-radius:var(--radius) !important; overflow:hidden; border:1px solid var(--border);
}
.section-card{
    background:var(--surface); border:1px solid var(--border); border-radius:var(--radius);
    padding:22px 26px; box-shadow:var(--shadow); margin:18px 0;
}

/* ---------------- Footer ---------------- */
.linkora-footer{
    margin-top:64px; padding:30px 10px; border-top:1px solid var(--border);
    text-align:center; color:var(--ink-soft); font-size:.83rem;
}
.linkora-footer .foot-links{ margin-bottom:10px; }
.linkora-footer .foot-links span{ margin:0 10px; font-weight:600; color:var(--ink); }

hr{ border-color:var(--border) !important; }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="linkora-nav">
        <div class="linkora-logo"><div class="mark">🛡️</div>Linkora</div>
        <div class="linkora-nav-links">
            <a href="#product">Product</a><a href="#how-it-works">How it works</a><a href="#docs">Docs</a>
        </div>
        <div class="linkora-badge">● Deep Learning Engine</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------------
# Hero
# --------------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero-wrap">
        <div class="hero-eyebrow">🔒 Real-time URL Risk Analysis</div>
        <h1 class="hero-title">Detect phishing URLs<br>before they reach your users.</h1>
        <p class="hero-sub">
            Linkora extracts 30 structural, WHOIS, and DNS signals from any URL and runs
            three deep learning models — ANN, CNN1D, and LSTM — side by side to give you
            a confident, explainable verdict.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------------
# How it works (always visible, above results)
# --------------------------------------------------------------------------------
st.markdown(
    """
    <div class="section-heading">
        <h2>How it works</h2>
        <p>From a raw URL to a trustworthy verdict in three steps.</p>
    </div>
    """,
    unsafe_allow_html=True,
)
how_cols = st.columns(3)
how_steps = [
    ("🔗", "Extract signals", "Linkora parses the URL and fetches page, DNS, and WHOIS "
     "data to compute 30 structural and behavioral features."),
    ("🧠", "Run 3 models", "ANN, CNN1D, and LSTM independently score the same feature "
     "vector so you can compare how each architecture reacts."),
    ("✅", "Get a verdict", "Results are combined into a majority-vote verdict with "
     "confidence scores, risk levels, and a recommendation for every model."),
]
for col, (icon, title, desc) in zip(how_cols, how_steps):
    with col:
        st.markdown(
            f"""
            <div class="step-card">
                <div class="step-num">{icon}</div>
                <h4>{title}</h4>
                <p>{desc}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

# --------------------------------------------------------------------------------
# Scan card / input (main focus of the page)
# --------------------------------------------------------------------------------
st.markdown('<div class="scan-card">', unsafe_allow_html=True)
st.markdown('<div class="scan-card-label">🔍 Scan a URL</div>', unsafe_allow_html=True)

col_input, col_button = st.columns([5, 1])
with col_input:
    url_input = st.text_input(
        "URL yang ingin diperiksa",
        placeholder="https://www.example.com/login",
        label_visibility="collapsed",
    )
with col_button:
    run_detection = st.button("Deteksi →", type="primary", width='stretch')

st.caption(
    "Proses mencakup pengambilan halaman & WHOIS domain, sehingga bisa memakan waktu "
    "beberapa detik tergantung kecepatan situs target."
)

# Settings, integrated inline instead of a sidebar
with st.expander("⚙️ Detection settings"):
    st.markdown('<div class="settings-label">Models to run & compare</div>', unsafe_allow_html=True)
    all_models = ["ANN", "CNN1D", "LSTM"]
    selected_models = st.multiselect(
        "Model yang dijalankan & dibandingkan:",
        options=all_models, default=all_models,
        label_visibility="collapsed",
    )
    c1, c2 = st.columns([1, 2])
    with c1:
        if st.button("🔄 Reset model cache", width='stretch'):
            st.cache_resource.clear()
            st.success("Cache dibersihkan. Model akan dimuat ulang saat deteksi berikutnya.")
    with c2:
        st.caption(
            "Models are trained on the UCI Phishing Websites Dataset. Some legacy features "
            "(e.g. Alexa Rank, public Google PageRank) rely on services that are no longer "
            "available and are approximated with heuristics — flagged transparently below."
        )

st.markdown('</div>', unsafe_allow_html=True)


# --------------------------------------------------------------------------------
# Jalankan deteksi
# --------------------------------------------------------------------------------
if run_detection:
    if not url_input.strip():
        st.warning("Masukkan URL terlebih dahulu.")
        st.stop()

    if not selected_models:
        st.warning("Pilih minimal satu model di pengaturan deteksi.")
        st.stop()

    # 1. Muat artefak (scaler, urutan fitur)
    try:
        scaler, feature_columns, model_info = load_artifacts()
    except Exception as e:
        st.error(f"❌ Gagal memuat artefak model: {e}")
        st.stop()

    # 2. Ekstrak fitur dari URL — premium multi-stage loading experience
    scan_box = st.empty()
    scan_steps = [
        "Menghubungi server & mengambil halaman",
        "Melakukan lookup DNS & WHOIS domain",
        "Mengekstrak 30 fitur numerik dari URL",
    ]

    def render_scan(active_idx, progress_pct):
        rows = ""
        for i, label in enumerate(scan_steps):
            state = "done" if i < active_idx else ("active" if i == active_idx else "")
            dot_icon = "✓" if state == "done" else ""
            rows += (
                f'<div class="scan-step {state}">'
                f'<div class="scan-dot"></div>{label} {dot_icon}</div>'
            )
        scan_box.markdown(
            f'<div class="scan-card scan-progress">'
            f'<div class="scan-progress-title"><div class="scan-spinner"></div>'
            f'Analyzing target…</div>{rows}'
            f'<div class="scan-bar-track"><div class="scan-bar-fill" style="width:{progress_pct}%;"></div></div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    render_scan(0, 10)
    time.sleep(0.25)
    render_scan(1, 45)
    t0 = time.time()
    try:
        extraction = extract_features(url_input)
    except Exception as e:
        scan_box.empty()
        st.error(f"❌ Gagal mengekstrak fitur dari URL: {e}")
        st.stop()
    extraction_time = time.time() - t0
    render_scan(2, 85)
    time.sleep(0.25)
    render_scan(3, 100)
    time.sleep(0.2)
    scan_box.empty()

    if not extraction.fetch_ok:
        st.warning(
            f"⚠️ Halaman tidak berhasil diakses langsung ({extraction.error}). "
            "Deteksi tetap dilanjutkan menggunakan fitur berbasis URL, WHOIS, dan DNS saja "
            "(fitur berbasis konten halaman di-set netral)."
        )

    if extraction.approximated_features:
        with st.expander(f"⚠️ {len(extraction.approximated_features)} fitur didekati secara heuristik (klik untuk detail)"):
            for item in extraction.approximated_features:
                st.markdown(f"- {item}")

    # 3. Muat model yang dipilih
    with st.spinner(f"Memuat {len(selected_models)} model…"):
        models_dict, load_errors = load_all_models(selected_models)

    for name, err in load_errors.items():
        st.error(f"❌ Gagal memuat model **{name}**: {err}")

    if not models_dict:
        st.stop()

    # 4. Prediksi
    try:
        results = predict_all(models_dict, extraction.features, feature_columns, scaler)
    except Exception as e:
        st.error(f"❌ Gagal menjalankan prediksi: {e}")
        st.stop()

    st.markdown(
        '<div class="section-heading" style="margin-top:40px;"><h2>📊 Detection Results</h2>'
        f'<p>Waktu ekstraksi fitur: {extraction_time:.2f}s · URL diperiksa: <code>{url_input}</code></p></div>',
        unsafe_allow_html=True,
    )

    # Verdict gabungan (voting mayoritas antar model yang berhasil)
    votes_phishing = sum(1 for r in results.values() if r["label"] == 1)
    votes_total = len(results)
    is_phishing = votes_phishing > votes_total / 2
    verdict_class = "verdict-danger" if is_phishing else "verdict-safe"
    verdict_icon = "🚩" if is_phishing else "✅"
    verdict_text = "Phishing detected" if is_phishing else "Legitimate — looks safe"

    st.markdown(
        f"""
        <div class="verdict-banner {verdict_class}">
            <div>
                <p class="verdict-title">{verdict_icon} {verdict_text}</p>
                <p class="verdict-sub">Majority vote across {votes_total} model(s) ·
                {votes_phishing}/{votes_total} flagged this URL as phishing</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Kartu hasil per model — circular progress rings + Risk Level + Recommendation
    cols = st.columns(len(results))
    for col, (name, r) in zip(cols, results.items()):
        with col:
            phishing_pct = r["prob_phishing"]
            danger = r["label"] == 1
            ring_color = "var(--danger)" if danger else "var(--safe)"
            pill_class = "pill-danger" if danger else "pill-safe"
            pill_text = "Phishing" if danger else "Aman"
            risk_label, recommendation = risk_profile(phishing_pct, r["label"])
            st.markdown(
                f"""
                <div class="model-card">
                    <div class="model-name">{name}</div>
                    <div class="ring" style="background:conic-gradient({ring_color} {phishing_pct * 3.6}deg, var(--border) 0deg);">
                        <div class="ring-inner">
                            <div class="ring-pct">{phishing_pct:.0f}%</div>
                            <div class="ring-caption">phishing score</div>
                        </div>
                    </div>
                    <span class="pill {pill_class}">{pill_text}</span>
                    <div class="model-conf">{r['confidence']:.2f}% keyakinan model</div>
                    <div class="card-divider"></div>
                    <div class="detail-row">
                        <span class="detail-key">Prediction</span>
                        <span class="detail-val">{r['label_text']}</span>
                    </div>
                    <div class="detail-row">
                        <span class="detail-key">Confidence</span>
                        <span class="detail-val">{r['confidence']:.1f}%</span>
                    </div>
                    <div class="detail-row">
                        <span class="detail-key">Risk Level</span>
                        <span class="risk-chip risk-{risk_label}">{risk_label}</span>
                    </div>
                    <div class="model-reco">{recommendation}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.write("")

    # Tabel perbandingan
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("#### Model comparison table")
    df_results = pd.DataFrame({
        name: {
            "Prediksi": r["label_text"],
            "Skor Phishing (%)": round(r["prob_phishing"], 2),
            "Skor Legitimate (%)": round(r["prob_legitimate"], 2),
            "Keyakinan (%)": round(r["confidence"], 2),
        }
        for name, r in results.items()
    }).T
    st.dataframe(df_results, width='stretch')
    st.markdown('</div>', unsafe_allow_html=True)

    # Grafik bar perbandingan skor phishing antar model
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("#### Phishing probability by model")
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=list(results.keys()),
        y=[r["prob_phishing"] for r in results.values()],
        marker_color=["#b0503b" if r["label"] == 1 else "#3f8556" for r in results.values()],
        text=[f"{r['prob_phishing']:.1f}%" for r in results.values()],
        textposition="outside",
        marker_line_width=0,
    ))
    fig.update_layout(
        yaxis_title="Probabilitas Phishing (%)",
        yaxis_range=[0, 110],
        height=360,
        margin=dict(t=20, l=10, r=10, b=10),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#3d2f22"),
    )
    st.plotly_chart(fig, width='stretch')
    st.markdown('</div>', unsafe_allow_html=True)

    if model_info.get("best_model") in results:
        best = model_info["best_model"]
        st.info(
            f"💡 Berdasarkan evaluasi di notebook, model dengan akurasi tertinggi pada "
            f"data uji adalah **{best}** — pertimbangkan hasil model ini sebagai acuan utama "
            f"jika prediksi antar model berbeda."
        )

    with st.expander("🔎 Lihat 30 fitur mentah hasil ekstraksi dari URL ini"):
        feat_df = pd.DataFrame({
            "Fitur": feature_columns,
            "Nilai (-1/0/1)": [extraction.features.get(c, 0) for c in feature_columns],
        })
        st.dataframe(feat_df, width='stretch', hide_index=True)

else:
    st.info("⬆️ Masukkan URL di atas lalu klik **Deteksi** untuk memulai analisis.")

# --------------------------------------------------------------------------------
# Footer
# --------------------------------------------------------------------------------
st.markdown(
    """
    <div class="linkora-footer">
        <div class="foot-links">
            <span>Linkora</span> · <span>Product</span> · <span>Docs</span> · <span>Privacy</span>
        </div>
        Built on the UCI Phishing Websites Dataset · ANN · CNN1D · LSTM ·
        Results are statistical estimates, not a guarantee — always use your own judgement.
    </div>
    """,
    unsafe_allow_html=True,
)