import streamlit as st
import pandas as pd
from recommender import (
    recommend_for_user,
    ratings,
    movies,
    get_user_rating_count,
    recommend_similar_movies,
)

st.set_page_config(
    page_title="Marquee",
    page_icon="🎞️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

if "page" not in st.session_state:
    st.session_state.page = "home"


def go(page: str):
    st.session_state.page = page
    st.rerun()


# ── CSS ──────────────────────────────────────────────────────────────────────
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Outfit:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Outfit', system-ui, sans-serif !important;
}
* { box-sizing: border-box; }
#MainMenu, footer, header { visibility: hidden !important; }
.stDeployButton { display: none !important; }
section[data-testid="stSidebar"] { display: none !important; }

.stApp { background: #06070e !important; }

.block-container {
    max-width: 1040px !important;
    padding: 2rem 3rem !important;
}

/* ── NAV BUTTONS (tagged via JS with data-marquee-nav) ── */
[data-marquee-nav="1"] .stButton > button {
    background: transparent !important;
    color: #3a4060 !important;
    border: 1px solid rgba(255,255,255,.07) !important;
    border-radius: 100px !important;
    padding: .28rem 1.1rem !important;
    font-size: .78rem !important;
    font-weight: 500 !important;
    font-family: 'Outfit', sans-serif !important;
    box-shadow: none !important;
    letter-spacing: .4px !important;
    line-height: 1.5 !important;
    transition: color .2s, border-color .2s, background .2s !important;
}
[data-marquee-nav="1"] .stButton > button:hover {
    background: rgba(99,102,241,.06) !important;
    color: #a5b4fc !important;
    border-color: rgba(99,102,241,.25) !important;
    box-shadow: none !important;
    transform: none !important;
}
/* Brand button */
[data-marquee-nav="1"] .stColumn:first-child .stButton > button {
    font-family: 'Bebas Neue', sans-serif !important;
    font-size: 1.45rem !important;
    letter-spacing: .1em !important;
    color: #e8ebff !important;
    border: none !important;
    padding: .15rem 0 !important;
    background: transparent !important;
}
[data-marquee-nav="1"] .stColumn:first-child .stButton > button:hover {
    color: #c7d2fe !important;
    background: transparent !important;
    border: none !important;
}
[data-marquee-nav="1"] .stColumn:first-child .stButton > button::after {
    content: '●';
    font-size: .4rem;
    color: #6366f1;
    text-shadow: 0 0 8px rgba(99,102,241,.9);
    margin-left: .38rem;
    vertical-align: middle;
    position: relative;
    top: -2px;
}

/* ── CTA BUTTONS ── */
.stButton > button {
    background: #6366f1 !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    padding: .65rem 1.8rem !important;
    font-weight: 600 !important;
    font-size: .88rem !important;
    font-family: 'Outfit', sans-serif !important;
    letter-spacing: .3px !important;
    transition: all .2s cubic-bezier(.4,0,.2,1) !important;
    box-shadow: 0 4px 24px rgba(99,102,241,.2) !important;
}
.stButton > button:hover {
    background: #4f46e5 !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 32px rgba(99,102,241,.32) !important;
}

/* ── NAV DIVIDER ── */
.nav-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(99,102,241,.1), transparent);
    margin: 0 0 3rem;
}

/* ── HERO ── */
.hero {
    text-align: center;
    padding: 4.5rem 0 3.5rem;
    position: relative;
}
.hero-glow {
    position: absolute;
    top: 10%; left: 50%;
    transform: translateX(-50%);
    width: 800px; height: 500px;
    background: radial-gradient(ellipse at center,
        rgba(99,102,241,.09) 0%,
        rgba(99,102,241,.03) 45%,
        transparent 70%);
    pointer-events: none;
}
.hero-eyebrow {
    display: inline-flex;
    align-items: center;
    gap: .5rem;
    background: rgba(99,102,241,.07);
    border: 1px solid rgba(99,102,241,.15);
    border-radius: 100px;
    padding: .3rem .9rem;
    font-size: .7rem;
    font-weight: 600;
    color: #6366f1;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-bottom: 1.6rem;
}
.hero-title {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 5.8rem;
    letter-spacing: .04em;
    line-height: .95;
    color: #eef0ff;
    margin: 0;
}
.hero-title-accent {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 5.8rem;
    letter-spacing: .04em;
    line-height: .95;
    background: linear-gradient(135deg, #a5b4fc 0%, #818cf8 55%, #6366f1 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    display: block;
    margin: 0 0 1.5rem;
}
.hero-sub {
    font-size: .95rem;
    color: #3d4461;
    max-width: 420px;
    margin: 0 auto 2.5rem;
    line-height: 1.8;
    font-weight: 400;
}
.hero-sub a { color: #6366f1; text-decoration: none; }

/* ── STATS STRIP ── */
.stats-strip {
    display: inline-flex;
    align-items: stretch;
    border: 1px solid rgba(255,255,255,.05);
    border-radius: 14px;
    background: rgba(255,255,255,.02);
    overflow: hidden;
    margin-bottom: 4rem;
}
.stat-item {
    padding: .75rem 1.5rem;
    text-align: center;
    border-right: 1px solid rgba(255,255,255,.04);
}
.stat-item:last-child { border-right: none; }
.stat-num {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1.4rem;
    letter-spacing: .06em;
    color: #c7d2fe;
    line-height: 1;
    display: block;
}
.stat-lbl {
    font-size: .64rem;
    color: #252c4a;
    text-transform: uppercase;
    letter-spacing: .9px;
    font-weight: 600;
    display: block;
    margin-top: .25rem;
}

/* ── FEATURE CARDS ── */
.feat-card {
    background: linear-gradient(155deg, #0e1022 0%, #090a16 100%);
    border: 1px solid rgba(255,255,255,.06);
    border-radius: 20px;
    padding: 2rem 1.8rem;
    position: relative;
    overflow: hidden;
    transition: border-color .3s, transform .35s cubic-bezier(.4,0,.2,1), box-shadow .35s;
}
.feat-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(99,102,241,.5), transparent);
}
.feat-card::after {
    content: '';
    position: absolute;
    bottom: -60px; right: -60px;
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(99,102,241,.05) 0%, transparent 70%);
    pointer-events: none;
}
.feat-card:hover {
    border-color: rgba(99,102,241,.22);
    transform: translateY(-6px);
    box-shadow: 0 28px 80px rgba(0,0,0,.5), 0 0 0 1px rgba(99,102,241,.08);
}
.feat-icon-wrap {
    width: 52px; height: 52px;
    background: rgba(99,102,241,.1);
    border: 1px solid rgba(99,102,241,.18);
    border-radius: 14px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.5rem;
    margin-bottom: 1.3rem;
}
.feat-title {
    font-size: 1.08rem;
    font-weight: 700;
    color: #eef0ff;
    margin: 0 0 .65rem;
    letter-spacing: -.01em;
}
.feat-desc {
    font-size: .83rem;
    color: #3d4461;
    line-height: 1.8;
    margin: 0 0 1.5rem;
}
.feat-tag {
    display: inline-flex;
    align-items: center;
    background: rgba(99,102,241,.07);
    border: 1px solid rgba(99,102,241,.12);
    border-radius: 8px;
    padding: .28rem .65rem;
    font-size: .7rem;
    font-weight: 600;
    color: rgba(99,102,241,.9);
    letter-spacing: .4px;
}

/* ── HOW IT WORKS ── */
.hiw {
    padding: 3.5rem 0 2rem;
    border-top: 1px solid rgba(255,255,255,.04);
    margin-top: .5rem;
    text-align: center;
}
.hiw-label {
    font-size: .68rem;
    font-weight: 600;
    color: #6366f1;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    margin-bottom: .5rem;
}
.hiw-title {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 2.8rem;
    letter-spacing: .06em;
    color: #eef0ff;
    margin: 0 0 .4rem;
}
.hiw-sub {
    font-size: .85rem;
    color: #252c4a;
    margin: 0 0 2.5rem;
}
.hiw-steps {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 1px;
    background: rgba(255,255,255,.04);
    border: 1px solid rgba(255,255,255,.04);
    border-radius: 16px;
    overflow: hidden;
    text-align: left;
}
.hiw-step {
    background: #07080e;
    padding: 2rem 1.5rem;
    transition: background .2s;
}
.hiw-step:hover { background: rgba(99,102,241,.03); }
.hiw-num {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 2.8rem;
    letter-spacing: .04em;
    color: rgba(99,102,241,.18);
    line-height: 1;
    margin-bottom: .75rem;
}
.hiw-step-title {
    font-size: .9rem;
    font-weight: 700;
    color: #c7d2fe;
    margin: 0 0 .4rem;
}
.hiw-step-desc {
    font-size: .78rem;
    color: #2d3460;
    line-height: 1.75;
    margin: 0;
}

/* ── PAGE HEADER ── */
.page-hdr {
    margin-bottom: 2rem;
    padding-bottom: 1.5rem;
    border-bottom: 1px solid rgba(255,255,255,.04);
}
.page-hdr h2 {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 2.4rem;
    letter-spacing: .07em;
    color: #eef0ff;
    margin: 0 0 .4rem;
}
.page-hdr p {
    font-size: .88rem;
    color: #3d4461;
    margin: 0;
    line-height: 1.75;
    max-width: 600px;
}

/* ── INFO BOX ── */
.info-box {
    background: rgba(99,102,241,.03);
    border: 1px solid rgba(99,102,241,.1);
    border-left: 3px solid rgba(99,102,241,.4);
    border-radius: 0 10px 10px 0;
    padding: .85rem 1.1rem;
    font-size: .82rem;
    color: #4a526e;
    line-height: 1.7;
    margin-bottom: 1.5rem;
}
.info-box strong { color: #a5b4fc; }
.info-box a { color: #6366f1; text-decoration: none; }

/* ── COLD START ── */
.cold-start {
    background: rgba(245,158,11,.03);
    border: 1px solid rgba(245,158,11,.1);
    border-left: 3px solid rgba(245,158,11,.35);
    border-radius: 0 10px 10px 0;
    padding: .65rem 1rem;
    font-size: .8rem;
    color: #b45309;
    display: flex;
    align-items: flex-start;
    gap: .5rem;
    margin: .5rem 0;
}

/* ── USER STATS ── */
.user-stats {
    display: flex;
    gap: .5rem;
    flex-wrap: wrap;
    margin: .8rem 0 1.4rem;
}
.u-stat {
    background: rgba(255,255,255,.02);
    border: 1px solid rgba(255,255,255,.05);
    border-radius: 8px;
    padding: .35rem .8rem;
    font-size: .78rem;
    color: #3d4461;
}
.u-stat strong { color: #c7d2fe; font-weight: 600; }

/* ── SECTION HEADER ── */
.sec-hdr {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1.4rem;
    letter-spacing: .08em;
    color: #818cf8;
    margin: 1.8rem 0 1rem;
    display: flex;
    align-items: center;
    gap: .5rem;
}

/* ── MOVIE LIST + CARDS ── */
.movie-list { display: flex; flex-direction: column; gap: .3rem; }
.movie-card {
    background: rgba(255,255,255,.02);
    border: 1px solid rgba(255,255,255,.04);
    border-radius: 12px;
    padding: .9rem 1.2rem .9rem 1rem;
    display: flex;
    align-items: center;
    gap: 1rem;
    transition: background .2s, border-color .2s, transform .2s;
    position: relative;
    overflow: hidden;
}
.movie-card::before {
    content: '';
    position: absolute;
    left: 0; top: 0; bottom: 0;
    width: 3px;
    background: linear-gradient(180deg, #6366f1, #4338ca);
    transform: scaleY(0);
    transform-origin: center;
    transition: transform .22s cubic-bezier(.4,0,.2,1);
    border-radius: 0 2px 2px 0;
}
.movie-card:hover {
    background: rgba(99,102,241,.04);
    border-color: rgba(99,102,241,.14);
    transform: translateX(3px);
}
.movie-card:hover::before { transform: scaleY(1); }
.movie-rank {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1.35rem;
    letter-spacing: .05em;
    color: rgba(99,102,241,.2);
    min-width: 28px;
    text-align: center;
    flex-shrink: 0;
    line-height: 1;
}
.movie-info { flex: 1; min-width: 0; }
.movie-title {
    font-size: .9rem;
    font-weight: 600;
    color: #dde1f5;
    margin: 0 0 .35rem;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    letter-spacing: -.01em;
}
.movie-genres { display: flex; flex-wrap: wrap; gap: .2rem; }
.genre-pill {
    background: rgba(99,102,241,.07);
    border: 1px solid rgba(99,102,241,.1);
    color: rgba(99,102,241,.75);
    font-size: .6rem;
    font-weight: 600;
    padding: .1rem .42rem;
    border-radius: 100px;
    letter-spacing: .4px;
    text-transform: uppercase;
}
.movie-score {
    text-align: right;
    min-width: 56px;
    flex-shrink: 0;
}
.score-val {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1.5rem;
    letter-spacing: .06em;
    color: #818cf8;
    line-height: 1;
    display: block;
}
.score-lbl {
    font-size: .6rem;
    color: #1e2240;
    text-transform: uppercase;
    letter-spacing: .8px;
    font-weight: 600;
    display: block;
    margin-top: .1rem;
}
.selected-card {
    background: rgba(99,102,241,.05) !important;
    border-color: rgba(99,102,241,.2) !important;
}

/* ── INPUTS ── */
div[data-baseweb="select"] > div {
    background: rgba(255,255,255,.03) !important;
    border-color: rgba(255,255,255,.07) !important;
    border-radius: 10px !important;
}
div[data-baseweb="select"] > div:hover {
    border-color: rgba(99,102,241,.3) !important;
}
.stSlider > div > div > div > div { background: #6366f1 !important; }
.stSelectbox label, .stSlider label {
    color: #2d3460 !important;
    font-size: .78rem !important;
    font-weight: 500 !important;
    letter-spacing: .2px !important;
}
.streamlit-expanderHeader {
    background: transparent !important;
    border: 1px solid rgba(255,255,255,.05) !important;
    border-radius: 10px !important;
    font-size: .8rem !important;
    color: #3d4461 !important;
}

/* ── SCROLLBAR ── */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(99,102,241,.15); border-radius: 2px; }

/* ── FOOTER ── */
.app-footer {
    text-align: center;
    padding: 2rem 0 .5rem;
    margin-top: 2.5rem;
    border-top: 1px solid rgba(255,255,255,.04);
    font-size: .72rem;
    color: #151929;
    letter-spacing: .3px;
}
.app-footer a { color: #1c2236; text-decoration: none; transition: color .2s; }
.app-footer a:hover { color: #6366f1; }
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)

# JS: tag the first stHorizontalBlock so CSS can target nav buttons reliably.
# Uses MutationObserver to survive React re-renders.
st.markdown(
    """
<script>
(function(){
    var A='data-marquee-nav';
    function tag(){
        var el=document.querySelector('[data-testid="stHorizontalBlock"]');
        if(el && !el.hasAttribute(A)){ el.setAttribute(A,'1'); }
    }
    tag();
    var ob=new MutationObserver(function(){ tag(); });
    ob.observe(document.body,{childList:true,subtree:true});
    setTimeout(function(){ ob.disconnect(); },4000);
})();
</script>
""",
    unsafe_allow_html=True,
)

# Active nav highlight
page = st.session_state.page
_active = {"collab": 2, "content": 3}
if page in _active:
    _col = _active[page]
    st.markdown(
        f"""
<style>
[data-marquee-nav="1"] .stColumn:nth-child({_col}) .stButton > button {{
    color: #818cf8 !important;
    border-color: rgba(99,102,241,.28) !important;
    background: rgba(99,102,241,.06) !important;
}}
</style>
""",
        unsafe_allow_html=True,
    )
else:
    # home active — brighten brand
    st.markdown(
        """
<style>
[data-marquee-nav="1"] .stColumn:first-child .stButton > button { color: #c7d2fe !important; }
</style>
""",
        unsafe_allow_html=True,
    )

# ── NAVIGATION ──────────────────────────────────────────────────────────────
nav_cols = st.columns([3.5, 1.4, 1.4])
with nav_cols[0]:
    if st.button("MARQUEE", key="nav_brand"):
        go("home")
with nav_cols[1]:
    if st.button("Collaborative", key="nav_collab", use_container_width=True):
        go("collab")
with nav_cols[2]:
    if st.button("Content-Based", key="nav_content", use_container_width=True):
        go("content")

st.markdown('<div class="nav-divider"></div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# HOME PAGE
# ═══════════════════════════════════════════════════════════════════════════════
if page == "home":
    st.markdown(
        """
<div class="hero">
    <div class="hero-glow"></div>
    <div class="hero-eyebrow">✦ ML-Powered Discovery</div>
    <div class="hero-title">Find Your Next</div>
    <div class="hero-title-accent">Favorite Film</div>
    <p class="hero-sub">
        Two machine learning algorithms trained on the
        <a href="https://grouplens.org/datasets/movielens/">MovieLens</a> dataset.
        Real recommendations, built from scratch.
    </p>
    <div class="stats-strip">
        <div class="stat-item">
            <span class="stat-num">100K</span>
            <span class="stat-lbl">Ratings</span>
        </div>
        <div class="stat-item">
            <span class="stat-num">9,742</span>
            <span class="stat-lbl">Movies</span>
        </div>
        <div class="stat-item">
            <span class="stat-num">610</span>
            <span class="stat-lbl">Users</span>
        </div>
        <div class="stat-item">
            <span class="stat-num">2</span>
            <span class="stat-lbl">Algorithms</span>
        </div>
    </div>
</div>
""",
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2, gap="medium")
    with c1:
        st.markdown(
            """
<div class="feat-card">
    <div class="feat-icon-wrap">👥</div>
    <div class="feat-title">Collaborative Filtering</div>
    <p class="feat-desc">Finds users with similar taste and predicts movies you'd rate highly.
    Built on a user×movie matrix with cosine similarity.</p>
    <span class="feat-tag">User-User KNN · RMSE 0.9764</span>
</div>
""",
            unsafe_allow_html=True,
        )
        if st.button("Try Collaborative →", key="go_collab", use_container_width=True):
            go("collab")

    with c2:
        st.markdown(
            """
<div class="feat-card">
    <div class="feat-icon-wrap">🎯</div>
    <div class="feat-title">Content-Based Filtering</div>
    <p class="feat-desc">Analyses genre DNA of movies using TF-IDF vectors. Finds the closest
    matches — no user ratings required.</p>
    <span class="feat-tag">TF-IDF + Cosine Sim · Genres only</span>
</div>
""",
            unsafe_allow_html=True,
        )
        if st.button("Try Content-Based →", key="go_content", use_container_width=True):
            go("content")

    st.markdown(
        """
<div class="hiw">
    <div class="hiw-label">Under the hood</div>
    <div class="hiw-title">How It Works</div>
    <p class="hiw-sub">From raw ratings data to personalised recommendations in three steps.</p>
    <div class="hiw-steps">
        <div class="hiw-step">
            <div class="hiw-num">01</div>
            <div class="hiw-step-title">Build the Matrix</div>
            <p class="hiw-step-desc">100K ratings from 610 users are transformed into a sparse user×movie matrix.</p>
        </div>
        <div class="hiw-step">
            <div class="hiw-num">02</div>
            <div class="hiw-step-title">Compute Similarity</div>
            <p class="hiw-step-desc">Cosine similarity measures how aligned two users' (or movies') vectors are.</p>
        </div>
        <div class="hiw-step">
            <div class="hiw-num">03</div>
            <div class="hiw-step-title">Predict & Rank</div>
            <p class="hiw-step-desc">Weighted average of top-K neighbours produces a predicted score for unseen movies.</p>
        </div>
    </div>
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown(
        """
<div class="app-footer">
    Built with Python · scikit-learn · Streamlit ·
    <a href="https://grouplens.org/datasets/movielens/">MovieLens Dataset</a>
</div>
""",
        unsafe_allow_html=True,
    )

# ═══════════════════════════════════════════════════════════════════════════════
# COLLABORATIVE FILTERING PAGE
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "collab":
    st.markdown(
        """
<div class="page-hdr">
    <h2>👥 Collaborative</h2>
    <p>Pick a user from the MovieLens dataset and discover movies predicted from the tastes of their most similar neighbours.</p>
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown(
        """
<div class="info-box">
    💡 Each <strong>User ID</strong> is a real, anonymised person from the
    <a href="https://grouplens.org/datasets/movielens/">MovieLens</a> research dataset.
    The model finds their closest neighbours and predicts scores for movies they haven't seen.
</div>
""",
        unsafe_allow_html=True,
    )

    col_sel, col_k, col_n = st.columns([2, 1, 1])
    with col_sel:
        user_ids = sorted(ratings["userId"].unique().tolist())
        user_id = st.selectbox(
            "Pick a user",
            user_ids,
            index=0,
            help="Each number is an anonymised person who has rated movies.",
        )
    with col_k:
        top_k = st.slider("Neighbors (K)", 3, 20, 5, help="Similar users to consider.")
    with col_n:
        n_recs = st.slider("Results", 5, 20, 10, help="How many recommendations.")

    rating_count = get_user_rating_count(user_id, ratings)
    user_ratings_df = (
        ratings[ratings["userId"] == user_id]
        .merge(movies, on="movieId")[["title", "genres", "rating"]]
        .sort_values("rating", ascending=False)
    )
    avg_rating = user_ratings_df["rating"].mean()
    fave = user_ratings_df.iloc[0]["title"]
    fave_short = fave if len(fave) <= 26 else fave[:23] + "…"

    st.markdown(
        f"""
<div class="user-stats">
    <div class="u-stat">🎬 <strong>{rating_count}</strong> rated</div>
    <div class="u-stat">⭐ avg <strong>{avg_rating:.1f}</strong></div>
    <div class="u-stat">🏆 <strong>{fave_short}</strong></div>
</div>
""",
        unsafe_allow_html=True,
    )

    if rating_count < 15:
        st.markdown(
            f"""
<div class="cold-start">
    <span style="flex-shrink:0;">⚠️</span>
    <span><strong>Cold-start:</strong> User {user_id} only has {rating_count} ratings —
    recommendations may be noisy.</span>
</div>
""",
            unsafe_allow_html=True,
        )

    with st.expander(f"📋 View all {rating_count} ratings"):
        st.dataframe(user_ratings_df.reset_index(drop=True), use_container_width=True, height=260)

    if st.button("🚀 Get Recommendations", key="collab_btn"):
        with st.spinner("Crunching similarities…"):
            try:
                recs = recommend_for_user(user_id, n=n_recs, top_k_users=top_k)
            except ValueError as e:
                st.error(str(e))
                st.stop()

        st.markdown(
            f'<div class="sec-hdr">🍿 Top {len(recs)} picks for User {user_id}</div>',
            unsafe_allow_html=True,
        )
        cards_html = '<div class="movie-list">'
        for i, rec in enumerate(recs, 1):
            g = "".join(
                f'<span class="genre-pill">{x.strip()}</span>'
                for x in rec["genres"].split("|")
            )
            cards_html += f"""
<div class="movie-card">
    <div class="movie-rank">{i:02d}</div>
    <div class="movie-info">
        <div class="movie-title">{rec['title']}</div>
        <div class="movie-genres">{g}</div>
    </div>
    <div class="movie-score">
        <span class="score-val">{rec['score']:.2f}</span>
        <span class="score-lbl">score</span>
    </div>
</div>"""
        cards_html += "</div>"
        st.markdown(cards_html, unsafe_allow_html=True)
        st.caption("Score = weighted avg rating from similar users. Higher = more loved.")

# ═══════════════════════════════════════════════════════════════════════════════
# CONTENT-BASED PAGE
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "content":
    st.markdown(
        """
<div class="page-hdr">
    <h2>🎯 Content-Based</h2>
    <p>Pick a movie you already enjoy and we'll find others with the most similar
    genre fingerprint using TF-IDF &amp; cosine similarity.</p>
</div>
""",
        unsafe_allow_html=True,
    )

    col_m, col_n2 = st.columns([3, 1])
    with col_m:
        movie_titles = sorted(movies["title"].unique().tolist())
        selected_movie_title = st.selectbox(
            "Search for a movie",
            movie_titles,
            index=0,
            help="Start typing to filter the catalogue.",
        )
    with col_n2:
        n_recs = st.slider("Results", 5, 20, 10, help="How many similar movies.")

    sel = movies[movies["title"] == selected_movie_title].iloc[0]
    sg = "".join(
        f'<span class="genre-pill">{x.strip()}</span>'
        for x in sel["genres"].split("|")
    )
    st.markdown(
        f"""<div class="movie-card selected-card">
    <div class="movie-rank">▶</div>
    <div class="movie-info">
        <div class="movie-title">{sel['title']}</div>
        <div class="movie-genres">{sg}</div>
    </div>
</div>""",
        unsafe_allow_html=True,
    )

    if st.button("🔍 Find Similar Movies", key="content_btn"):
        with st.spinner("Matching genre vectors…"):
            try:
                sim_recs = recommend_similar_movies(sel["movieId"], n=n_recs)
            except Exception as e:
                st.error(str(e))
                st.stop()

        st.markdown(
            f'<div class="sec-hdr">🍿 Similar to: {selected_movie_title}</div>',
            unsafe_allow_html=True,
        )
        cards_html = '<div class="movie-list">'
        for i, rec in enumerate(sim_recs, 1):
            g = "".join(
                f'<span class="genre-pill">{x.strip()}</span>'
                for x in rec["genres"].split("|")
            )
            cards_html += f"""
<div class="movie-card">
    <div class="movie-rank">{i:02d}</div>
    <div class="movie-info">
        <div class="movie-title">{rec['title']}</div>
        <div class="movie-genres">{g}</div>
    </div>
    <div class="movie-score">
        <span class="score-val">{rec['score']:.2f}</span>
        <span class="score-lbl">sim</span>
    </div>
</div>"""
        cards_html += "</div>"
        st.markdown(cards_html, unsafe_allow_html=True)
        st.caption("Similarity = cosine similarity between TF-IDF genre vectors. 1.0 = identical.")
