import streamlit as st
import pandas as pd
from recommender import (
    recommend_for_user,
    ratings,
    movies,
    get_user_rating_count,
    recommend_similar_movies,
)

# ── page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── session state ────────────────────────────────────────────────────────────
if "page" not in st.session_state:
    st.session_state.page = "home"


def go(page: str):
    st.session_state.page = page


# ── CSS ──────────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

/* ─── reset ─── */
html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}
#MainMenu, footer, header { visibility: hidden !important; }
.stDeployButton { display: none !important; }
section[data-testid="stSidebar"] { display: none !important; }

.stApp { background: #060611 !important; }

/* ─── container ─── */
.app-container {
    max-width: 900px;
    margin: 0 auto;
    padding: 0 1rem;
}

/* ─── top nav ─── */
.topnav {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 1rem 0 1.5rem;
    border-bottom: 1px solid rgba(99,102,241,.08);
    margin-bottom: 2rem;
}
.topnav-brand {
    font-size: 1.1rem;
    font-weight: 700;
    color: #e2e8f0;
    display: flex;
    align-items: center;
    gap: .45rem;
    letter-spacing: -.01em;
}
.topnav-links {
    display: flex;
    gap: .3rem;
}
.topnav-link {
    padding: .4rem .9rem;
    border-radius: 8px;
    font-size: .78rem;
    font-weight: 500;
    color: #64748b;
    text-decoration: none;
    transition: all .2s ease;
    cursor: pointer;
    border: 1px solid transparent;
}
.topnav-link:hover {
    color: #a5b4fc;
    background: rgba(99,102,241,.06);
}
.topnav-link.active {
    color: #c7d2fe;
    background: rgba(99,102,241,.1);
    border-color: rgba(99,102,241,.2);
}

/* ─── hero (home) ─── */
.home-hero {
    text-align: center;
    padding: 4rem 0 3rem;
}
.home-hero-title {
    font-size: 3.2rem;
    font-weight: 900;
    letter-spacing: -.04em;
    line-height: 1.1;
    background: linear-gradient(135deg, #e2e8f0 0%, #a5b4fc 50%, #818cf8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 1rem;
}
.home-hero-sub {
    font-size: 1.05rem;
    color: #64748b;
    max-width: 520px;
    margin: 0 auto 1.5rem;
    line-height: 1.7;
    font-weight: 400;
}
.home-stats {
    display: flex;
    justify-content: center;
    gap: .6rem;
    flex-wrap: wrap;
    margin-bottom: 3rem;
}
.home-stat {
    background: rgba(99,102,241,.05);
    border: 1px solid rgba(99,102,241,.1);
    border-radius: 10px;
    padding: .45rem .85rem;
    font-size: .72rem;
    font-weight: 600;
    color: #818cf8;
    letter-spacing: .3px;
}

/* ─── feature cards ─── */
.features {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1.2rem;
    margin-bottom: 3rem;
}
.feature-card {
    background: rgba(12,12,24,.6);
    border: 1px solid rgba(99,102,241,.08);
    border-radius: 18px;
    padding: 2rem 1.8rem;
    transition: all .3s cubic-bezier(.4,0,.2,1);
    position: relative;
    overflow: hidden;
}
.feature-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, rgba(99,102,241,.3), transparent);
    opacity: 0;
    transition: opacity .3s ease;
}
.feature-card:hover {
    border-color: rgba(99,102,241,.2);
    transform: translateY(-3px);
    box-shadow: 0 12px 40px rgba(99,102,241,.08);
}
.feature-card:hover::before {
    opacity: 1;
}
.feature-icon {
    font-size: 2.2rem;
    margin-bottom: 1rem;
}
.feature-title {
    font-size: 1.15rem;
    font-weight: 700;
    color: #e2e8f0;
    margin: 0 0 .55rem;
}
.feature-desc {
    font-size: .82rem;
    color: #64748b;
    line-height: 1.65;
    margin: 0 0 1.2rem;
}
.feature-meta {
    font-size: .68rem;
    color: #475569;
    font-weight: 500;
    letter-spacing: .3px;
}
.feature-meta strong {
    color: #818cf8;
}

/* ─── how it works section ─── */
.how-section {
    text-align: center;
    padding: 2rem 0 3rem;
    border-top: 1px solid rgba(99,102,241,.06);
}
.how-title {
    font-size: 1.4rem;
    font-weight: 800;
    color: #e2e8f0;
    margin: 0 0 .5rem;
    letter-spacing: -.02em;
}
.how-sub {
    font-size: .85rem;
    color: #475569;
    margin: 0 0 2rem;
}
.how-steps {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 1rem;
}
.how-step {
    background: rgba(12,12,24,.4);
    border: 1px solid rgba(99,102,241,.06);
    border-radius: 14px;
    padding: 1.4rem 1.2rem;
    text-align: center;
}
.how-step-num {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 32px; height: 32px;
    border-radius: 50%;
    background: rgba(99,102,241,.08);
    border: 1px solid rgba(99,102,241,.15);
    color: #818cf8;
    font-size: .75rem;
    font-weight: 700;
    margin-bottom: .75rem;
}
.how-step-title {
    font-size: .85rem;
    font-weight: 600;
    color: #c7d2fe;
    margin: 0 0 .3rem;
}
.how-step-desc {
    font-size: .72rem;
    color: #475569;
    line-height: 1.55;
    margin: 0;
}

/* ─── page header ─── */
.page-hdr {
    margin-bottom: 1.5rem;
}
.page-hdr h2 {
    font-size: 1.5rem;
    font-weight: 800;
    color: #e2e8f0;
    margin: 0 0 .3rem;
    letter-spacing: -.02em;
}
.page-hdr p {
    font-size: .85rem;
    color: #64748b;
    margin: 0;
    line-height: 1.6;
}

/* ─── info box ─── */
.info-box {
    background: rgba(99,102,241,.04);
    border: 1px solid rgba(99,102,241,.1);
    border-radius: 12px;
    padding: .75rem .95rem;
    font-size: .78rem;
    color: #8892b0;
    line-height: 1.6;
    margin-bottom: 1.2rem;
}
.info-box strong { color: #a5b4fc; }
.info-box a { color: #818cf8; text-decoration: none; }

/* ─── cold start ─── */
.cold-start {
    background: rgba(251,191,36,.04);
    border: 1px solid rgba(251,191,36,.12);
    border-radius: 12px;
    padding: .7rem .9rem;
    margin: .6rem 0;
    font-size: .78rem;
    color: #fbbf24;
    display: flex;
    align-items: flex-start;
    gap: .5rem;
}

/* ─── stat pills ─── */
.stats-row {
    display: flex;
    gap: .4rem;
    flex-wrap: wrap;
    margin: .5rem 0;
}
.stat-pill {
    background: rgba(15,15,28,.6);
    border: 1px solid rgba(99,102,241,.1);
    border-radius: 8px;
    padding: .3rem .6rem;
    font-size: .72rem;
    color: #8892b0;
}
.stat-pill strong { color: #e2e8f0; font-weight: 600; }

/* ─── movie card ─── */
.movie-card {
    background: rgba(12,12,24,.5);
    border: 1px solid rgba(99,102,241,.06);
    border-radius: 14px;
    padding: 1rem 1.2rem;
    margin-bottom: .5rem;
    display: flex;
    align-items: center;
    gap: 1rem;
    transition: all .25s cubic-bezier(.4,0,.2,1);
}
.movie-card:hover {
    border-color: rgba(99,102,241,.18);
    background: rgba(18,18,36,.6);
    transform: translateY(-1px);
    box-shadow: 0 6px 24px rgba(0,0,0,.2);
}
.movie-rank {
    font-size: 1.1rem;
    font-weight: 800;
    color: rgba(100,116,139,.25);
    min-width: 28px;
    text-align: center;
}
.movie-info { flex: 1; min-width: 0; }
.movie-title {
    font-size: .88rem;
    font-weight: 600;
    color: #e2e8f0;
    margin: 0 0 .3rem;
}
.movie-genres { display: flex; flex-wrap: wrap; gap: .25rem; }
.genre-pill {
    background: rgba(99,102,241,.06);
    border: 1px solid rgba(99,102,241,.12);
    color: #818cf8;
    font-size: .6rem;
    font-weight: 500;
    padding: .12rem .42rem;
    border-radius: 100px;
}
.movie-score { text-align: right; min-width: 52px; }
.score-val {
    font-size: 1.2rem;
    font-weight: 700;
    color: #818cf8;
}
.score-lbl {
    font-size: .55rem;
    color: #475569;
    text-transform: uppercase;
    letter-spacing: .7px;
}

/* ─── selected card ─── */
.selected-card {
    border-color: rgba(99,102,241,.18) !important;
    background: rgba(99,102,241,.04) !important;
}

/* ─── section header ─── */
.sec-hdr {
    font-size: 1rem;
    font-weight: 700;
    color: #a5b4fc;
    margin: 1.2rem 0 .7rem;
    display: flex;
    align-items: center;
    gap: .4rem;
}

/* ─── buttons ─── */
.stButton > button {
    background: linear-gradient(135deg, #6366f1, #4f46e5) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 12px !important;
    padding: .6rem 2rem !important;
    font-weight: 600 !important;
    font-size: .85rem !important;
    transition: all .25s cubic-bezier(.4,0,.2,1) !important;
    box-shadow: 0 4px 16px rgba(99,102,241,.2) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 28px rgba(99,102,241,.3) !important;
}

/* ─── selectbox ─── */
div[data-baseweb="select"] > div {
    background: rgba(12,12,24,.6) !important;
    border-color: rgba(99,102,241,.1) !important;
    border-radius: 10px !important;
}
div[data-baseweb="select"] > div:hover {
    border-color: rgba(99,102,241,.25) !important;
}

/* ─── slider ─── */
.stSlider > div > div > div > div {
    background: #6366f1 !important;
}

/* ─── expander ─── */
.streamlit-expanderHeader {
    background: transparent !important;
    border: 1px solid rgba(99,102,241,.06) !important;
    border-radius: 10px !important;
    font-size: .8rem !important;
    color: #64748b !important;
}

/* ─── back button ─── */
.back-btn {
    display: inline-flex;
    align-items: center;
    gap: .35rem;
    padding: .4rem .85rem;
    border-radius: 8px;
    font-size: .78rem;
    font-weight: 500;
    color: #64748b;
    background: rgba(99,102,241,.04);
    border: 1px solid rgba(99,102,241,.08);
    cursor: pointer;
    transition: all .2s ease;
    text-decoration: none;
    margin-bottom: 1.5rem;
}
.back-btn:hover {
    color: #a5b4fc;
    border-color: rgba(99,102,241,.2);
    background: rgba(99,102,241,.08);
}

/* ─── scrollbar ─── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(99,102,241,.12); border-radius: 3px; }

/* ─── footer ─── */
.app-footer {
    text-align: center;
    padding: 2rem 0;
    margin-top: 2rem;
    border-top: 1px solid rgba(99,102,241,.06);
    font-size: .7rem;
    color: #334155;
}
.app-footer a { color: #475569; text-decoration: none; }
.app-footer a:hover { color: #818cf8; }
</style>
""",
    unsafe_allow_html=True,
)

# ═══════════════════════════════════════════════════════════════════════════════
# NAVIGATION
# ═══════════════════════════════════════════════════════════════════════════════
page = st.session_state.page
nav_home = "active" if page == "home" else ""
nav_collab = "active" if page == "collab" else ""
nav_content = "active" if page == "content" else ""

# We use columns + buttons for navigation since HTML links can't trigger session state
nav_cols = st.columns([3, 1, 1, 1])
with nav_cols[0]:
    st.markdown('<div class="topnav-brand">🎬 Movie Recommender</div>', unsafe_allow_html=True)
with nav_cols[1]:
    if st.button("Home", key="nav_home", use_container_width=True):
        go("home")
        st.rerun()
with nav_cols[2]:
    if st.button("Collaborative", key="nav_collab", use_container_width=True):
        go("collab")
        st.rerun()
with nav_cols[3]:
    if st.button("Content-Based", key="nav_content", use_container_width=True):
        go("content")
        st.rerun()

st.markdown("<div style='height:1px;background:rgba(99,102,241,.06);margin-bottom:2rem;'></div>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# HOME PAGE
# ═══════════════════════════════════════════════════════════════════════════════
if page == "home":
    st.markdown(
        """
<div class="home-hero">
    <div class="home-hero-title">Discover Your<br>Next Favorite Movie</div>
    <p class="home-hero-sub">
        A recommendation engine built from scratch with two ML algorithms.
        Powered by the <a href="https://grouplens.org/datasets/movielens/" style="color:#818cf8;">MovieLens</a> dataset.
    </p>
    <div class="home-stats">
        <span class="home-stat">📊 100,836 ratings</span>
        <span class="home-stat">🎥 9,742 movies</span>
        <span class="home-stat">👤 610 users</span>
        <span class="home-stat">🧠 2 algorithms</span>
    </div>
</div>
""",
        unsafe_allow_html=True,
    )

    # feature cards
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            """
<div class="feature-card">
    <div class="feature-icon">👥</div>
    <div class="feature-title">Collaborative Filtering</div>
    <p class="feature-desc">
        Finds users with similar taste and predicts what you'd rate highly.
        Built on a user×movie matrix with cosine similarity.
    </p>
    <div class="feature-meta">
        Method: <strong>User-User KNN</strong> · RMSE: <strong>0.9764</strong>
    </div>
</div>
""",
            unsafe_allow_html=True,
        )
        if st.button("Try Collaborative →", key="go_collab", use_container_width=True):
            go("collab")
            st.rerun()

    with c2:
        st.markdown(
            """
<div class="feature-card">
    <div class="feature-icon">🎯</div>
    <div class="feature-title">Content-Based Filtering</div>
    <p class="feature-desc">
        Analyses genre DNA of movies using TF-IDF vectors. Finds the closest
        matches — no user ratings required.
    </p>
    <div class="feature-meta">
        Method: <strong>TF-IDF + Cosine Sim</strong> · Genres only
    </div>
</div>
""",
            unsafe_allow_html=True,
        )
        if st.button("Try Content-Based →", key="go_content", use_container_width=True):
            go("content")
            st.rerun()

    # how it works
    st.markdown(
        """
<div class="how-section">
    <div class="how-title">How It Works</div>
    <p class="how-sub">From raw data to personalized recommendations in three steps.</p>
    <div class="how-steps">
        <div class="how-step">
            <div class="how-step-num">1</div>
            <div class="how-step-title">Build the Matrix</div>
            <p class="how-step-desc">
                100K ratings from 610 users are transformed into a sparse user×movie matrix.
            </p>
        </div>
        <div class="how-step">
            <div class="how-step-num">2</div>
            <div class="how-step-title">Compute Similarity</div>
            <p class="how-step-desc">
                Cosine similarity measures how aligned two users' (or movies') vectors are.
            </p>
        </div>
        <div class="how-step">
            <div class="how-step-num">3</div>
            <div class="how-step-title">Predict & Rank</div>
            <p class="how-step-desc">
                Weighted average of top-K neighbors produces a predicted score for unseen movies.
            </p>
        </div>
    </div>
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown(
        """
<div class="app-footer">
    Built with Python, scikit-learn &amp; Streamlit ·
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
    <h2>👥 Collaborative Filtering</h2>
    <p>Pick a user from the MovieLens dataset and discover movies predicted
    from the tastes of their most similar users.</p>
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown(
        """
<div class="info-box">
    💡 Each <strong>User ID</strong> is a real, anonymised person from the
    <a href="https://grouplens.org/datasets/movielens/">MovieLens</a> research dataset.
    They've each rated dozens to hundreds of movies. The model finds
    similar users and predicts new movies they'd enjoy.
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
<div class="stats-row">
    <div class="stat-pill">🎬 <strong>{rating_count}</strong> rated</div>
    <div class="stat-pill">⭐ <strong>{avg_rating:.1f}</strong> avg</div>
    <div class="stat-pill">🏆 <strong>{fave_short}</strong></div>
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
        for i, rec in enumerate(recs, 1):
            g = "".join(f'<span class="genre-pill">{x.strip()}</span>' for x in rec["genres"].split("|"))
            st.markdown(
                f"""<div class="movie-card">
    <div class="movie-rank">{i}</div>
    <div class="movie-info">
        <div class="movie-title">{rec['title']}</div>
        <div class="movie-genres">{g}</div>
    </div>
    <div class="movie-score">
        <div class="score-val">{rec['score']:.2f}</div>
        <div class="score-lbl">score</div>
    </div>
</div>""",
                unsafe_allow_html=True,
            )
        st.caption("Score = weighted avg rating from similar users. Higher = more loved.")

# ═══════════════════════════════════════════════════════════════════════════════
# CONTENT-BASED PAGE
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "content":
    st.markdown(
        """
<div class="page-hdr">
    <h2>🎯 Content-Based Filtering</h2>
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
    sg = "".join(f'<span class="genre-pill">{x.strip()}</span>' for x in sel["genres"].split("|"))
    st.markdown(
        f"""<div class="movie-card selected-card">
    <div class="movie-rank">🎥</div>
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
        for i, rec in enumerate(sim_recs, 1):
            g = "".join(f'<span class="genre-pill">{x.strip()}</span>' for x in rec["genres"].split("|"))
            st.markdown(
                f"""<div class="movie-card">
    <div class="movie-rank">{i}</div>
    <div class="movie-info">
        <div class="movie-title">{rec['title']}</div>
        <div class="movie-genres">{g}</div>
    </div>
    <div class="movie-score">
        <div class="score-val">{rec['score']:.2f}</div>
        <div class="score-lbl">similarity</div>
    </div>
</div>""",
                unsafe_allow_html=True,
            )
        st.caption("Similarity = cosine similarity between TF-IDF genre vectors. 1.0 = identical.")
