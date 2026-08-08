"""
app.py
-------
HotelScope — a dark-themed, storytelling Streamlit dashboard for the
"Investigate Hotel Business using Data Visualization" project.

Pages (sidebar navigation):
  Home                  -> hero + auto-generated executive briefing
  Executive Dashboard   -> KPI cards + hotel-type overview
  Booking Trends        -> seasonality / monthly booking patterns
  Cancellation Analysis -> stay duration & lead time vs cancellation
  Revenue & Customers   -> ADR trends, market segment, customer type
  Recommendations       -> business recommendations + data download

Run with:
    streamlit run app.py
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import streamlit.components.v1 as components

from data_cleaning import load_raw_data, clean_data, RAW_PATH, CLEAN_PATH

st.set_page_config(page_title="HotelScope | Booking Intelligence", layout="wide", page_icon="🏨")

# ----------------------------------------------------------------------
# Dark theme colour palette (shared by CSS and Matplotlib)
# ----------------------------------------------------------------------
BG_DARK = "#0b0e1a"
BG_CARD = "#12162a"
ACCENT_CYAN = "#4fd1ff"
ACCENT_PURPLE = "#b388ff"
ACCENT_ORANGE = "#ffb86b"
TEXT_LIGHT = "#e8e8f5"
TEXT_DIM = "#9aa0c0"
GRID_COLOR = "#2a2f4a"

PALETTE = [ACCENT_CYAN, ACCENT_ORANGE]


def style_dark(fig, ax):
    """Apply the dashboard's dark theme to a Matplotlib figure/axis."""
    fig.patch.set_facecolor(BG_CARD)
    ax.set_facecolor(BG_CARD)
    ax.tick_params(colors=TEXT_DIM, labelsize=9)
    ax.xaxis.label.set_color(TEXT_DIM)
    ax.yaxis.label.set_color(TEXT_DIM)
    ax.title.set_color(TEXT_LIGHT)
    for spine in ax.spines.values():
        spine.set_color(GRID_COLOR)
    ax.grid(True, color=GRID_COLOR, linewidth=0.6, alpha=0.6)
    legend = ax.get_legend()
    if legend:
        legend.get_frame().set_facecolor(BG_CARD)
        legend.get_frame().set_edgecolor(GRID_COLOR)
        for text in legend.get_texts():
            text.set_color(TEXT_LIGHT)
        if legend.get_title():
            legend.get_title().set_color(TEXT_LIGHT)
    return fig, ax


sns.set_style("darkgrid", {
    "axes.facecolor": BG_CARD, "figure.facecolor": BG_CARD,
    "grid.color": GRID_COLOR, "text.color": TEXT_LIGHT,
})

# ----------------------------------------------------------------------
# Global CSS: dark theme, hero, nav pills, cards, animations
# ----------------------------------------------------------------------
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {{ font-family: 'Poppins', sans-serif; }}

[data-testid="stAppViewContainer"], [data-testid="stHeader"] {{
    background-color: {BG_DARK};
}}
[data-testid="stAppViewContainer"] > .main {{
    animation: pageFadeIn 0.7s ease both;
}}
@keyframes pageFadeIn {{
    from {{ opacity: 0; transform: translateY(10px); }}
    to   {{ opacity: 1; transform: translateY(0); }}
}}

/* Sidebar */
[data-testid="stSidebar"] {{
    background-color: #090c17;
    border-right: 1px solid {GRID_COLOR};
}}
[data-testid="stSidebar"] * {{ color: {TEXT_LIGHT}; }}

.logo-card {{
    background: linear-gradient(160deg, #141936, #0d1024);
    border: 1px solid {GRID_COLOR};
    border-radius: 16px;
    padding: 22px 18px;
    margin-bottom: 18px;
    text-align: center;
    animation: cardFadeUp 0.6s ease both;
}}
.logo-card h2 {{
    margin: 0; font-size: 1.4rem; font-weight: 800; color: {TEXT_LIGHT};
}}
.logo-card p {{
    margin: 6px 0 0 0; font-size: 0.78rem; color: {TEXT_DIM}; font-style: italic;
}}
.logo-card .tag {{
    margin-top: 10px; font-size: 0.65rem; letter-spacing: 0.08em;
    color: {ACCENT_CYAN}; text-transform: uppercase;
}}

/* Sidebar nav buttons -> pill style */
[data-testid="stSidebar"] .stButton > button {{
    width: 100%; text-align: left; background: transparent;
    border: 1px solid transparent; border-radius: 10px;
    padding: 10px 14px; font-size: 0.92rem; font-weight: 500;
    color: {TEXT_DIM}; transition: all 0.2s ease; margin-bottom: 4px;
}}
[data-testid="stSidebar"] .stButton > button:hover {{
    background: rgba(79,209,255,0.08); border-color: {GRID_COLOR}; color: {TEXT_LIGHT};
    transform: translateX(3px);
}}
[data-testid="stSidebar"] .stButton > button[kind="primary"] {{
    background: linear-gradient(90deg, rgba(79,209,255,0.18), rgba(179,136,255,0.18));
    border: 1px solid {ACCENT_CYAN}; color: {TEXT_LIGHT}; font-weight: 600;
    box-shadow: 0 0 14px rgba(79,209,255,0.15);
}}

/* Hero section */
.hero-box {{
    background: radial-gradient(circle at 20% 20%, #1a1f3d 0%, #0c0f1f 70%);
    border: 1px solid {GRID_COLOR};
    border-radius: 20px;
    padding: 40px 44px;
    margin-bottom: 26px;
    animation: cardFadeUp 0.7s ease both;
}}
.hero-badge {{
    display: inline-block; border: 1px solid {ACCENT_CYAN}; color: {ACCENT_CYAN};
    border-radius: 999px; padding: 6px 16px; font-size: 0.72rem;
    letter-spacing: 0.08em; text-transform: uppercase; margin-bottom: 18px;
}}
.hero-title {{
    font-size: 3.6rem; font-weight: 800; margin: 0 0 14px 0; line-height: 1.05;
    background: linear-gradient(90deg, {ACCENT_CYAN}, {ACCENT_PURPLE});
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text;
    animation: fadeSlideDown 0.9s ease both;
}}
.hero-sub {{
    color: {TEXT_DIM}; font-size: 1.05rem; max-width: 760px; line-height: 1.6;
    margin-bottom: 20px; animation: fadeSlideDown 1.1s ease both;
}}
@keyframes fadeSlideDown {{
    from {{ opacity: 0; transform: translateY(-10px); }}
    to   {{ opacity: 1; transform: translateY(0); }}
}}
.welcome-line {{
    font-size: 1.15rem; color: {TEXT_LIGHT}; font-weight: 500;
    border-left: 3px solid {ACCENT_CYAN}; padding-left: 14px; margin: 18px 0 22px 0;
    min-height: 1.6em;
}}
.badge-row {{ display: flex; gap: 12px; flex-wrap: wrap; margin-top: 6px; }}
.badge-pill {{
    background: rgba(255,255,255,0.04); border: 1px solid {GRID_COLOR};
    border-radius: 10px; padding: 10px 16px; font-size: 0.85rem; color: {TEXT_LIGHT};
    transition: all 0.25s ease; animation: cardFadeUp 0.8s ease both;
}}
.badge-pill:hover {{
    border-color: {ACCENT_CYAN}; transform: translateY(-3px);
    box-shadow: 0 8px 20px rgba(79,209,255,0.12);
}}

/* Generic dark card */
.dark-card {{
    background: {BG_CARD}; border: 1px solid {GRID_COLOR}; border-radius: 16px;
    padding: 24px 26px; margin-bottom: 20px; animation: cardFadeUp 0.7s ease both;
}}
.dark-card h3 {{ color: {TEXT_LIGHT}; margin-top: 0; }}
.dark-card p {{ color: {TEXT_DIM}; line-height: 1.7; font-size: 0.98rem; }}
.dark-card .eyebrow {{
    display: inline-block; border: 1px solid {ACCENT_PURPLE}; color: {ACCENT_PURPLE};
    border-radius: 999px; padding: 4px 14px; font-size: 0.68rem;
    letter-spacing: 0.08em; text-transform: uppercase; margin-bottom: 14px;
}}
@keyframes cardFadeUp {{
    from {{ opacity: 0; transform: translateY(14px); }}
    to   {{ opacity: 1; transform: translateY(0); }}
}}

/* KPI cards */
.kpi-row {{ display: flex; gap: 16px; margin-bottom: 24px; flex-wrap: wrap; }}
.kpi-card {{
    flex: 1; min-width: 190px; background: {BG_CARD}; border: 1px solid {GRID_COLOR};
    border-radius: 14px; padding: 18px 20px; border-left: 4px solid {ACCENT_CYAN};
    transition: transform 0.25s ease, box-shadow 0.25s ease; animation: cardFadeUp 0.6s ease both;
}}
.kpi-card:hover {{ transform: translateY(-6px); box-shadow: 0 10px 26px rgba(79,209,255,0.12); }}
.kpi-card:nth-child(1) {{ animation-delay: 0.05s; border-left-color: {ACCENT_CYAN}; }}
.kpi-card:nth-child(2) {{ animation-delay: 0.15s; border-left-color: {ACCENT_ORANGE}; }}
.kpi-card:nth-child(3) {{ animation-delay: 0.25s; border-left-color: {ACCENT_PURPLE}; }}
.kpi-card:nth-child(4) {{ animation-delay: 0.35s; border-left-color: #6ee7b7; }}
.kpi-label {{ font-size: 0.75rem; color: {TEXT_DIM}; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; }}
.kpi-value {{ font-size: 1.75rem; font-weight: 700; color: {TEXT_LIGHT}; margin-top: 4px; }}

h1, h2, h3, h4, p, span, label {{ color: {TEXT_LIGHT}; }}
[data-testid="stImage"] img {{ transition: transform 0.3s ease; border-radius: 12px; }}
[data-testid="stImage"] img:hover {{ transform: scale(1.01); }}
[data-testid="stDataFrame"] {{ border-radius: 10px; overflow: hidden; }}
.stSpinner > div > div {{ border-top-color: {ACCENT_CYAN} !important; }}
hr {{ border-color: {GRID_COLOR}; }}

/* ============ Intro storytelling overlay ============ */
.intro-overlay {{
    position: fixed; inset: 0; z-index: 9999;
    background: radial-gradient(circle at 50% 38%, #1a1f3d 0%, #05060d 82%);
    display: flex; flex-direction: column; align-items: center; justify-content: center;
    animation: introFadeOut 6.5s ease forwards;
}}
@keyframes introFadeOut {{
    0%, 85% {{ opacity: 1; visibility: visible; }}
    100%    {{ opacity: 0; visibility: hidden; }}
}}
.intro-scene {{ position: relative; width: 460px; max-width: 88vw; height: 210px; margin-bottom: 26px; }}
.intro-door {{
    position: absolute; right: 16px; top: 14px; font-size: 56px;
    filter: drop-shadow(0 0 18px rgba(79,209,255,0.4));
}}
.walker {{
    position: absolute; font-size: 32px; opacity: 0;
    animation: walkIn 1.9s ease forwards;
}}
.walker.w1 {{ left: -40px; bottom: 130px; animation-delay: 0.3s; }}
.walker.w2 {{ left: -40px; bottom: 68px;  animation-delay: 1.1s; font-size: 28px; }}
.walker.w3 {{ left: -40px; bottom: 6px;   animation-delay: 1.9s; font-size: 36px; }}
@keyframes walkIn {{
    0%   {{ opacity: 0; transform: translateX(0); }}
    12%  {{ opacity: 1; }}
    100% {{ opacity: 1; transform: translateX(330px); }}
}}
.bubble {{
    position: absolute; background: rgba(255,255,255,0.07);
    border: 1px solid {ACCENT_CYAN}; border-radius: 12px; padding: 6px 14px;
    font-size: 0.78rem; color: {TEXT_LIGHT}; opacity: 0; white-space: nowrap;
    animation: bubbleShow 0.9s ease forwards;
}}
.bubble.b1 {{ left: 190px; bottom: 172px; animation-delay: 2.0s; }}
.bubble.b2 {{ left: 210px; bottom: 108px; animation-delay: 2.8s; }}
.bubble.b3 {{ left: 170px; bottom: 46px;  animation-delay: 3.6s; }}
@keyframes bubbleShow {{
    from {{ opacity: 0; transform: translateY(6px); }}
    to   {{ opacity: 1; transform: translateY(0); }}
}}
.intro-title {{
    font-size: 2.1rem; font-weight: 800; opacity: 0; text-align: center;
    background: linear-gradient(90deg, {ACCENT_CYAN}, {ACCENT_PURPLE});
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
    animation: introTitleShow 1s ease forwards; animation-delay: 4.4s;
}}
.intro-sub {{
    color: {TEXT_DIM}; font-size: 0.9rem; opacity: 0; margin-top: 8px; text-align: center;
    animation: introTitleShow 1s ease forwards; animation-delay: 4.8s;
}}
@keyframes introTitleShow {{
    from {{ opacity: 0; transform: translateY(10px); }}
    to   {{ opacity: 1; transform: translateY(0); }}
}}

/* ============ Live moving cursor (decorative) ============ */
.live-cursor-wrap {{ position: relative; }}
.cursor-track {{ position: relative; height: 4px; margin: 4px 0 22px 0; }}
.live-cursor {{
    position: absolute; width: 14px; height: 14px; border-radius: 50%;
    background: radial-gradient(circle, {ACCENT_CYAN} 0%, rgba(79,209,255,0) 70%);
    box-shadow: 0 0 12px 4px rgba(79,209,255,0.45);
    animation: cursorMove 9s ease-in-out infinite;
    top: -8px; pointer-events: none; z-index: 5;
}}
@keyframes cursorMove {{
    0%, 18%  {{ left: 8%;  top: -8px; }}
    25%, 43% {{ left: 33%; top: -8px; }}
    50%, 68% {{ left: 58%; top: -8px; }}
    75%, 93% {{ left: 82%; top: -8px; }}
    100%     {{ left: 8%;  top: -8px; }}
}}
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------
# One-time storytelling intro: three guests walk in and welcome you,
# then the overlay fades away to reveal the dashboard.
# ----------------------------------------------------------------------
if "intro_played" not in st.session_state:
    st.session_state.intro_played = True
    st.markdown("""
    <div class="intro-overlay">
        <div class="intro-scene">
            <div class="intro-door">🚪</div>
            <div class="walker w1">🧍</div>
            <div class="bubble b1">Welcome!</div>
            <div class="walker w2">🧳</div>
            <div class="bubble b2">Welcome to the hotel!</div>
            <div class="walker w3">🧑‍🤝‍🧑</div>
            <div class="bubble b3">Enjoy your stay!</div>
        </div>
        <div class="intro-title">Welcome to HotelScope</div>
        <div class="intro-sub">Preparing your booking intelligence dashboard…</div>
    </div>
    """, unsafe_allow_html=True)


# ----------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------
@st.cache_data
def get_data() -> pd.DataFrame:
    if os.path.exists(CLEAN_PATH):
        df = pd.read_csv(CLEAN_PATH)
    else:
        df_raw = load_raw_data(RAW_PATH)
        df = clean_data(df_raw)
        os.makedirs("data", exist_ok=True)
        df.to_csv(CLEAN_PATH, index=False)

    month_order = ["January", "February", "March", "April", "May", "June",
                   "July", "August", "September", "October", "November", "December"]
    df["arrival_date_month"] = pd.Categorical(df["arrival_date_month"], categories=month_order, ordered=True)
    stay_order = ["1 night", "2-3 nights", "4-7 nights", "8-14 nights", "15+ nights"]
    df["stay_duration_bucket"] = pd.Categorical(df["stay_duration_bucket"], categories=stay_order, ordered=True)
    lead_order = ["0-7 days", "8-30 days", "31-90 days", "91-180 days", "181-365 days", "365+ days"]
    df["lead_time_bucket"] = pd.Categorical(df["lead_time_bucket"], categories=lead_order, ordered=True)
    return df


with st.spinner("🏨 Loading and preparing booking data..."):
    df = get_data()

# ----------------------------------------------------------------------
# Sidebar: logo, navigation, filters
# ----------------------------------------------------------------------
PAGES = [
    ("🏠", "Home"),
    ("📊", "Executive Dashboard"),
    ("📅", "Booking Trends"),
    ("❌", "Cancellation Analysis"),
    ("💰", "Revenue & Customers"),
    ("💡", "Recommendations"),
]

if "page" not in st.session_state:
    st.session_state.page = "Home"

with st.sidebar:
    st.markdown("""
    <div class="logo-card">
        <h2>🏨 HotelScope</h2>
        <p>Where hospitality meets intelligence.</p>
        <div class="tag">Booking Intelligence Platform</div>
    </div>
    """, unsafe_allow_html=True)

    for icon, name in PAGES:
        is_active = st.session_state.page == name
        if st.button(f"{icon}   {name}", key=f"nav_{name}", use_container_width=True,
                     type="primary" if is_active else "secondary"):
            st.session_state.page = name
            st.rerun()

    st.markdown("---")
    with st.expander("🔍 Filters", expanded=False):
        hotel_options = ["All"] + sorted(df["hotel"].unique().tolist())
        hotel_filter = st.selectbox("Hotel type", hotel_options)
        year_options = sorted(df["arrival_date_year"].unique().tolist())
        year_filter = st.multiselect("Arrival year", year_options, default=year_options)
        status_options = ["All", "Canceled", "Not Canceled"]
        status_filter = st.selectbox("Booking status", status_options)

    st.markdown("---")
    st.caption(f"Dataset: {len(df):,} cleaned bookings · {df['arrival_date_year'].min()}–{df['arrival_date_year'].max()}")
    st.markdown(
        '<div style="font-style:italic; color:#9aa0c0; font-size:0.8rem; margin-top:10px;">'
        '"Turning every reservation into a clearer business decision."</div>',
        unsafe_allow_html=True,
    )

filtered = df.copy()
if hotel_filter != "All":
    filtered = filtered[filtered["hotel"] == hotel_filter]
if year_filter:
    filtered = filtered[filtered["arrival_date_year"].isin(year_filter)]
if status_filter != "All":
    filtered = filtered[filtered["is_canceled_label"] == status_filter]

if filtered.empty:
    st.warning("No bookings match the current filters. Please broaden your selection in the sidebar.")
    st.stop()


# ----------------------------------------------------------------------
# Reusable KPI row
# ----------------------------------------------------------------------
def render_kpis(data, key_prefix=""):
    kpis = [
        {"label": "Total Bookings", "value": len(data), "decimals": 0, "prefix": "", "suffix": ""},
        {"label": "Cancellation Rate", "value": data["is_canceled"].mean() * 100, "decimals": 1, "prefix": "", "suffix": "%"},
        {"label": "Avg. Lead Time", "value": data["lead_time"].mean(), "decimals": 0, "prefix": "", "suffix": " days"},
        {"label": "Avg. Daily Rate", "value": data["adr"].mean(), "decimals": 2, "prefix": "$", "suffix": ""},
    ]
    html = '<div class="kpi-row live-cursor-wrap"><div class="live-cursor"></div>'
    for i, k in enumerate(kpis):
        html += f"""
        <div class="kpi-card">
            <div class="kpi-label">{k['label']}</div>
            <div class="kpi-value">
                <span class="counter-{key_prefix}" data-target="{k['value']:.4f}" data-decimals="{k['decimals']}"
                      data-prefix="{k['prefix']}" data-suffix="{k['suffix']}">0</span>
            </div>
        </div>"""
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)
    components.html(f"""
    <script>
    setTimeout(function() {{
        const counters = window.parent.document.querySelectorAll('.counter-{key_prefix}');
        counters.forEach(function(counter) {{
            if (counter.dataset.animated === "1") return;
            counter.dataset.animated = "1";
            const target = parseFloat(counter.getAttribute('data-target'));
            const decimals = parseInt(counter.getAttribute('data-decimals'));
            const prefix = counter.getAttribute('data-prefix') || '';
            const suffix = counter.getAttribute('data-suffix') || '';
            const duration = 900;
            const startTime = performance.now();
            function animate(time) {{
                const progress = Math.min((time - startTime) / duration, 1);
                const eased = 1 - Math.pow(1 - progress, 3);
                const current = target * eased;
                counter.textContent = prefix + current.toLocaleString('en-US', {{
                    minimumFractionDigits: decimals, maximumFractionDigits: decimals
                }}) + suffix;
                if (progress < 1) requestAnimationFrame(animate);
            }}
            requestAnimationFrame(animate);
        }});
    }}, 50);
    </script>
    """, height=0)


def typewriter(text, elem_id, speed=28):
    st.markdown(f'<div class="welcome-line" id="{elem_id}"></div>', unsafe_allow_html=True)
    safe_text = text.replace("\\", "\\\\").replace("`", "\\`")
    components.html(f"""
    <script>
    setTimeout(function() {{
        const el = window.parent.document.getElementById("{elem_id}");
        if (!el || el.dataset.typed === "1") return;
        el.dataset.typed = "1";
        const text = `{safe_text}`;
        let i = 0;
        function type() {{
            if (i <= text.length) {{
                el.textContent = text.slice(0, i);
                i++;
                setTimeout(type, {speed});
            }}
        }}
        type();
    }}, 150);
    </script>
    """, height=0)


def cursor_track():
    """A thin decorative strip with a glowing dot that drifts left-to-right,
    giving each page a sense of live movement without overlapping content."""
    st.markdown(
        '<div class="cursor-track live-cursor-wrap"><div class="live-cursor"></div></div>',
        unsafe_allow_html=True,
    )


# ----------------------------------------------------------------------
# PAGE: Home
# ----------------------------------------------------------------------
if st.session_state.page == "Home":
    st.markdown("""
    <div class="hero-box">
        <div class="hero-badge">◆ Hospitality Business Intelligence</div>
        <div class="hero-title">HOTELSCOPE</div>
        <div class="hero-sub">A booking intelligence platform that turns raw hotel reservation data into
        strategic business insight — helping management understand demand, seasonality, and cancellation
        behaviour at a glance.</div>
    </div>
    """, unsafe_allow_html=True)

    dominant_hotel = filtered["hotel"].value_counts().idxmax()
    dominant_share = filtered["hotel"].value_counts(normalize=True).max() * 100
    welcome_msg = (
        f"Welcome back. {len(filtered):,} bookings are currently in view, "
        f"led by {dominant_hotel} at {dominant_share:.0f}% of the selection. "
        f"Here's what's happening across your hotels today."
    )
    typewriter(welcome_msg, "welcome-home")

    st.markdown('<div class="badge-row">', unsafe_allow_html=True)
    for label in ["📊 Interactive Analytics", "📈 Seasonal Trends", "❌ Cancellation Intelligence", "💰 Revenue Signals"]:
        st.markdown(f'<span class="badge-pill">{label}</span>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    render_kpis(filtered, key_prefix="home")

    avg_stay = filtered["total_stay_nights"].mean()
    top_month = filtered["arrival_date_month"].value_counts().idxmax()
    briefing = (
        f"The selected portfolio contains <b>{len(filtered):,}</b> valid bookings. "
        f"The current cancellation rate is <b>{filtered['is_canceled'].mean()*100:.1f}%</b>, while average stay "
        f"length is <b>{avg_stay:.1f} nights</b> and average lead time is <b>{filtered['lead_time'].mean():.0f} days</b>. "
        f"<b>{top_month}</b> is the busiest arrival month in the current selection. "
        f"{dominant_hotel} continues to account for the larger share of bookings at <b>{dominant_share:.0f}%</b>."
    )
    st.markdown(f"""
    <div class="dark-card">
        <div class="eyebrow">◆ Executive Briefing</div>
        <h3>Manager's Note</h3>
        <p>{briefing}</p>
    </div>
    """, unsafe_allow_html=True)

    st.caption("Use the navigation panel on the left to explore booking trends, cancellation patterns, revenue, and recommendations in detail.")


# ----------------------------------------------------------------------
# PAGE: Executive Dashboard
# ----------------------------------------------------------------------
elif st.session_state.page == "Executive Dashboard":
    st.markdown('<h2>📊 Executive Dashboard</h2>', unsafe_allow_html=True)
    st.caption("A snapshot of overall booking volume, cancellation rate, and hotel-type mix for the current selection.")
    cursor_track()
    render_kpis(filtered, key_prefix="exec")

    c1, c2 = st.columns([1, 2])
    with c1:
        counts = filtered["hotel"].value_counts()
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.pie(
            counts, labels=counts.index, autopct="%1.1f%%", colors=PALETTE, startangle=90,
            textprops={"color": TEXT_LIGHT},
        )
        ax.set_title("Share of Bookings by Hotel Type")
        style_dark(fig, ax)
        st.pyplot(fig)

    with c2:
        cancel_by_hotel = filtered.groupby("hotel")["is_canceled"].mean().mul(100)
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.bar(cancel_by_hotel.index, cancel_by_hotel.values, color=PALETTE)
        for i, v in enumerate(cancel_by_hotel.values):
            ax.text(i, v + 0.8, f"{v:.1f}%", ha="center", color=TEXT_LIGHT)
        ax.set_title("Overall Cancellation Rate by Hotel Type")
        ax.set_ylabel("Cancellation Rate (%)")
        style_dark(fig, ax)
        st.pyplot(fig)


# ----------------------------------------------------------------------
# PAGE: Booking Trends
# ----------------------------------------------------------------------
elif st.session_state.page == "Booking Trends":
    st.markdown('<h2>📅 Booking Trends</h2>', unsafe_allow_html=True)
    st.caption("Which hotel type is booked most often, and how does demand shift across the year?")
    cursor_track()

    monthly = filtered.groupby(["arrival_date_month", "hotel"], observed=True).size().reset_index(name="bookings")
    fig, ax = plt.subplots(figsize=(12, 5))
    for hotel_name, color in zip(sorted(monthly["hotel"].unique()), PALETTE):
        sub = monthly[monthly["hotel"] == hotel_name]
        ax.plot(sub["arrival_date_month"], sub["bookings"], marker="o", label=hotel_name, color=color)
    ax.set_title("Monthly Bookings by Hotel Type")
    ax.set_xlabel("Month")
    ax.set_ylabel("Number of Bookings")
    ax.tick_params(axis="x", rotation=45)
    ax.legend(title="Hotel")
    style_dark(fig, ax)
    st.pyplot(fig)
    st.caption("Bookings peak in the early autumn months and dip during the winter/early-spring period.")


# ----------------------------------------------------------------------
# PAGE: Cancellation Analysis
# ----------------------------------------------------------------------
elif st.session_state.page == "Cancellation Analysis":
    st.markdown('<h2>❌ Cancellation Analysis</h2>', unsafe_allow_html=True)
    st.caption("Does length of stay or lead time influence how often a booking gets cancelled?")
    cursor_track()

    st.markdown("#### Stay duration vs. cancellation")
    stay_cancel = (
        filtered.groupby(["stay_duration_bucket", "hotel"], observed=True)["is_canceled"]
        .mean().mul(100).reset_index(name="cancel_rate")
    )
    fig, ax = plt.subplots(figsize=(11, 5))
    width = 0.35
    buckets = stay_cancel["stay_duration_bucket"].cat.categories
    x = np.arange(len(buckets))
    for i, (hotel_name, color) in enumerate(zip(sorted(stay_cancel["hotel"].unique()), PALETTE)):
        sub = stay_cancel[stay_cancel["hotel"] == hotel_name].set_index("stay_duration_bucket").reindex(buckets)
        ax.bar(x + (i - 0.5) * width, sub["cancel_rate"], width, label=hotel_name, color=color)
    ax.set_xticks(x)
    ax.set_xticklabels(buckets, rotation=15)
    ax.set_title("Cancellation Rate by Length of Stay")
    ax.set_ylabel("Cancellation Rate (%)")
    ax.legend(title="Hotel")
    style_dark(fig, ax)
    st.pyplot(fig)

    st.markdown("#### Lead time vs. cancellation")
    lead_cancel = (
        filtered.groupby(["lead_time_bucket", "hotel"], observed=True)["is_canceled"]
        .mean().mul(100).reset_index(name="cancel_rate")
    )
    fig, ax = plt.subplots(figsize=(11, 5))
    for hotel_name, color in zip(sorted(lead_cancel["hotel"].unique()), PALETTE):
        sub = lead_cancel[lead_cancel["hotel"] == hotel_name]
        ax.plot(sub["lead_time_bucket"], sub["cancel_rate"], marker="o", label=hotel_name, color=color)
    ax.set_title("Cancellation Rate by Lead Time")
    ax.set_xlabel("Lead Time Bucket")
    ax.set_ylabel("Cancellation Rate (%)")
    ax.legend(title="Hotel")
    style_dark(fig, ax)
    st.pyplot(fig)
    st.caption("Cancellation risk climbs steadily with both longer stays and longer lead times — most sharply for City Hotel.")


# ----------------------------------------------------------------------
# PAGE: Revenue & Customers
# ----------------------------------------------------------------------
elif st.session_state.page == "Revenue & Customers":
    st.markdown('<h2>💰 Revenue & Customers</h2>', unsafe_allow_html=True)
    st.caption("Average daily rate trends and the customer segments driving bookings.")
    cursor_track()

    c1, c2 = st.columns(2)
    with c1:
        adr_trend = filtered.groupby(["arrival_date_month", "hotel"], observed=True)["adr"].mean().reset_index()
        fig, ax = plt.subplots(figsize=(6, 5))
        for hotel_name, color in zip(sorted(adr_trend["hotel"].unique()), PALETTE):
            sub = adr_trend[adr_trend["hotel"] == hotel_name]
            ax.plot(sub["arrival_date_month"], sub["adr"], marker="o", label=hotel_name, color=color)
        ax.set_title("Average Daily Rate by Month")
        ax.set_ylabel("ADR ($)")
        ax.tick_params(axis="x", rotation=60)
        ax.legend(title="Hotel")
        style_dark(fig, ax)
        st.pyplot(fig)

    with c2:
        seg = filtered.groupby("market_segment")["is_canceled"].mean().mul(100).sort_values(ascending=False).head(6)
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.barh(seg.index[::-1], seg.values[::-1], color=ACCENT_PURPLE)
        ax.set_title("Cancellation Rate by Market Segment")
        ax.set_xlabel("Cancellation Rate (%)")
        style_dark(fig, ax)
        st.pyplot(fig)

    total_rev = (filtered.loc[filtered["is_canceled"] == 0, "adr"] *
                 filtered.loc[filtered["is_canceled"] == 0, "total_stay_nights"]).sum()
    repeat_pct = filtered["is_repeated_guest"].mean() * 100
    st.markdown(f"""
    <div class="dark-card">
        <div class="eyebrow">◆ Revenue Snapshot</div>
        <p><b>Estimated realized revenue</b> (completed stays only, ADR × nights): <b>${total_rev:,.0f}</b><br>
        <b>Repeat guests:</b> {repeat_pct:.1f}% of bookings in the current selection</p>
    </div>
    """, unsafe_allow_html=True)


# ----------------------------------------------------------------------
# PAGE: Recommendations
# ----------------------------------------------------------------------
elif st.session_state.page == "Recommendations":
    st.markdown('<h2>💡 Recommendations</h2>', unsafe_allow_html=True)
    st.caption("Actionable next steps derived directly from the patterns observed in this dataset.")

    recs = [
        ("Hotel type & seasonality", "City Hotel drives the bulk of bookings; run targeted off-season promotions for Resort Hotel and add staffing/inventory ahead of the autumn peak for both properties."),
        ("Stay duration", "Since cancellations rise sharply with longer stays, consider tiered deposit or cancellation-fee policies for stays beyond ~7 nights, especially for City Hotel."),
        ("Lead time", "Since far-ahead bookings cancel more often, add pre-arrival reminder emails, partial non-refundable deposits, or free-rescheduling options for bookings made 90+ days out."),
    ]
    for title, body in recs:
        st.markdown(f"""
        <div class="dark-card">
            <div class="eyebrow">◆ Recommendation</div>
            <h3>{title}</h3>
            <p>{body}</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("#### Cleaned dataset preview")
    st.dataframe(filtered.head(200), use_container_width=True)

    st.download_button(
        "⬇ Download cleaned dataset (CSV)",
        data=filtered.to_csv(index=False).encode("utf-8"),
        file_name="hotel_bookings_clean_filtered.csv",
        mime="text/csv",
    )