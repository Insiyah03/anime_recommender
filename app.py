import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from recommenders import (
    get_top_animes, get_most_popular_animes, get_top_genres,
    get_cbr, search_by_synopsis, give_associations,
    filter_animes, surprise_me
)

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Anime Recommender",
    page_icon="🍜",
    layout="wide"
)


# ── Card Button CSS ───────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Fix all anime card buttons to the same height */
div[data-testid="stButton"] button[kind="secondary"] {
    height: 40 px !important;
    overflow: hidden !important;
    white-space: normal !important;
    display: -webkit-box !important;
    -webkit-line-clamp: 2 !important;
    -webkit-box-orient: vertical !important;
    text-overflow: ellipsis !important;
    line-height: 1.2 !important;
    font-size: 0.78rem !important;
    padding: 4px 6px !important;
    align-items: center !important;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────

# ── Load Data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    animes = pd.read_parquet('data/animes.parquet')
    embeddings = np.load('models/embeddings.npy')
    rules = pd.read_pickle('models/rules.pkl')
    return animes, embeddings, rules

@st.cache_resource
def load_model():
    return SentenceTransformer('all-mpnet-base-v2')

animes, embeddings, rules = load_data()
model = load_model()

# ── Constants ─────────────────────────────────────────────────────────────────
PLACEHOLDER_IMG = "https://stanleymovietheater.com/wp-content/uploads/woocommerce-placeholder.png"

# ── Helper Functions ──────────────────────────────────────────────────────────
import requests

def is_valid_image(url):
    try:
        response = requests.head(url, timeout=3)
        return response.status_code == 200
    except:
        return False

def show_image(url, width=150):
    if pd.isna(url) or url == '' or not is_valid_image(url):
        st.image(PLACEHOLDER_IMG, width=width)
    else:
        st.image(url, width=width)

def navigate_to_detail(uid):
    """Save current page and navigate to detail view."""
    st.session_state['prev_page'] = st.session_state.get('page', 'home')
    st.session_state['detail_uid'] = uid
    st.session_state['page'] = 'detail'



def show_anime_card(anime, key_suffix=""):
    """
    Reusable anime card: image, title, score, and a clickable button to open detail page.
    Call this inside a st.columns() block.

    Usage:
        cols = st.columns(7)
        for i, (_, anime) in enumerate(df.iterrows()):
            with cols[i % 7]:
                show_anime_card(anime, key_suffix=f"section_{i}")
    """
    img_url = anime.get('img_url', '')
    if pd.isna(img_url) or img_url == '' or not is_valid_image(img_url):
        st.image(PLACEHOLDER_IMG, width=150)
    else:
        st.image(img_url, width=150)

    uid = anime.get('uid', anime['title'])  # fall back to title if no uid
    btn_key = f"card_btn_{uid}_{key_suffix}"
    if st.button(f"{anime['title']}", key=btn_key, use_container_width=True):
        navigate_to_detail(uid)
        st.rerun()


def show_surprise_result(key='surprise'):
    if key in st.session_state and st.session_state[key] is not None:
        anime = st.session_state[key]
        col1, col2 = st.columns([1, 3])
        with col1:
            show_image(anime['img_url'])
        with col2:
            st.subheader(anime['title'])
            st.write(f"⭐ {anime['score']} | 👥 {anime['members']:,} | 🎬 {anime['episodes']} eps")
            st.write(f"🎭 {anime['genre']}")
            st.write(anime['synopsis'])
            uid = anime.get('uid', anime['title'])
            if st.button("ℹ️ Full Details", key=f"surprise_detail_{key}"):
                navigate_to_detail(uid)
                st.rerun()

# ── Navigation ────────────────────────────────────────────────────────────────
if 'page' not in st.session_state:
    st.session_state['page'] = 'home'


with st.sidebar:
    st.title("☰ Menu")
    st.divider()
    if st.button("🏠 Home", use_container_width=True):
        st.session_state['prev_page'] = st.session_state['page']
        st.session_state['page'] = 'home'
    if st.button("🎯 Content Based", use_container_width=True):
        st.session_state['prev_page'] = st.session_state['page']
        st.session_state['page'] = 'cbr'
    if st.button("🔍 Search", use_container_width=True):
        st.session_state['prev_page'] = st.session_state['page']
        st.session_state['page'] = 'search'

page = st.session_state['page']

# ── Search Bar ────────────────────────────────────────────────────────────────
def anime_search_bar(key, show_label=True):
    query = st.text_input(
        "🔍 Search Anime",  # always a real label
        placeholder="Type a title...",
        key=key,
        label_visibility="visible" if show_label else "collapsed"
    )
    if query:
        st.session_state['search_query'] = query
        st.session_state['prev_page'] = st.session_state.get('page', 'home')
        st.session_state['page'] = 'results'
        st.rerun()

# ── Carousel Rows ─────────────────────────────────────────────────────────────
import streamlit.components.v1 as components

def show_carousel(title, anime_df):
    anime_list = anime_df.to_dict('records')
    looped = anime_list * 3

    cards_html = ""
    for anime in looped:
        img = anime.get('img_url', '') or ''
        if pd.isna(img) or img == '':
            img = PLACEHOLDER_IMG
        title_text = str(anime['title']).replace("'", "\\'").replace('"', '&quot;')
        score = anime.get('score', 'N/A')

        cards_html += f"""
        <div class="anime-card">
            <img src="{img}" onerror="this.src='{PLACEHOLDER_IMG}'" loading="lazy"/>
            <div class="anime-title">{title_text}</div>
        </div>
        """

    uid = title.replace(' ', '').replace('🏆','').replace('🔥','').replace('🎭','')

    html = f"""
    <div class="carousel-wrapper">
        <div class="carousel-track" id="track-{uid}">
            {cards_html}
        </div>
    </div>

    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{ background: transparent; }}
        .carousel-wrapper {{ margin-bottom: 8px; font-family: 'Segoe UI', sans-serif; }}
        .carousel-track {{
            display: flex; overflow-x: auto; gap: 1px; padding-bottom: 6px;
            scroll-behavior: auto; -webkit-overflow-scrolling: touch; cursor: grab;
        }}
        .carousel-track:active {{ cursor: grabbing; }}
        .carousel-track::-webkit-scrollbar {{ height: 9px; }}
        .carousel-track::-webkit-scrollbar-thumb {{ background: #00000; border-radius: 10px; }}
        .carousel-track::-webkit-scrollbar-track {{ background: transparent; }}
        .anime-card {{
            min-width: 140px; max-width: 140px; flex-shrink: 0; border-radius: 6px;
            overflow: hidden; position: relative; transition: transform 0.25s ease; background: #111;
        }}
        .anime-card:hover {{ transform: scale(1.08); z-index: 10; }}
        .anime-card img {{ width: 100%; height: 200px; object-fit: cover; display: block; }}
        .anime-title {{
            color: #f0f0f0; font-size: 0.72rem; font-weight: 600; padding: 5px 6px 6px;
            background: #1a1a1a; white-space: nowrap; overflow: hidden;
            text-overflow: ellipsis; width: 100%;
        }}
    </style>

    <script>
        (function() {{
            const track = document.getElementById("track-{uid}");
            if (!track) return;
            const totalItems = {len(anime_list)};
            const cardWidth = 150;
            track.scrollLeft = totalItems * cardWidth;
            track.addEventListener('scroll', function() {{
                const max = track.scrollWidth - track.clientWidth;
                const segmentWidth = totalItems * cardWidth;
                if (track.scrollLeft <= segmentWidth * 0.5) track.scrollLeft += segmentWidth;
                if (track.scrollLeft >= max - segmentWidth * 0.5) track.scrollLeft -= segmentWidth;
            }});
            let isDown = false, startX, scrollStart;
            track.addEventListener('mousedown', e => {{
                isDown = true; startX = e.pageX - track.offsetLeft;
                scrollStart = track.scrollLeft; track.style.scrollBehavior = 'auto';
            }});
            track.addEventListener('mouseleave', () => isDown = false);
            track.addEventListener('mouseup', () => isDown = false);
            track.addEventListener('mousemove', e => {{
                if (!isDown) return; e.preventDefault();
                const x = e.pageX - track.offsetLeft;
                track.scrollLeft = scrollStart - (x - startX);
            }});
        }})();
    </script>
    """
    components.html(html, height=270, scrolling=False)


# ── Pages ─────────────────────────────────────────────────────────────────────
if page == 'home':
    st.title("🍜 Anime Recommender")
    st.divider()

    col1, col2 = st.columns([3, 1])
    with col1:
        anime_search_bar(key="home_search")
    with col2:
        st.write("")
        st.write("")
        if st.button("🎲 Surprise Me!", key="home_surprise"):
            st.session_state['surprise_home'] = surprise_me(animes)

    show_surprise_result(key='surprise_home')

    shown_uids = set()

    st.header("🏆 Top Rated")
    top = get_top_animes(animes, top_n=14)
    shown_uids.update(top['uid'].tolist())
    show_carousel("🏆 Top Rated", top)

    st.header("🔥 Most Popular")
    popular = get_most_popular_animes(animes, top_n=30)
    popular = popular[~popular['uid'].isin(shown_uids)].head(14)
    shown_uids.update(popular['uid'].tolist())
    show_carousel("🔥 Most Popular", popular)

    st.header("🎭 Browse by Genre")
    genres = ['Action', 'Comedy', 'Romance', 'Horror', 'Sci-Fi', 'Fantasy']
    for genre in genres:
        st.subheader(genre)
        top_genre = get_top_genres(animes, genre, top_n=30)
        top_genre = top_genre[~top_genre['uid'].isin(shown_uids)].head(14)
        shown_uids.update(top_genre['uid'].tolist())
        show_carousel(f"{genre}", top_genre)

elif page == 'cbr':
    left, right = st.columns([2, 1])
    with left:
        st.title("🎯 Find Similar Anime")
    with right:
        col1, col2 = st.columns([1, 1])
        with col1:
            st.write("")
            anime_search_bar(key="cbr_search", show_label=False)
        with col2:
            st.write("")
            if st.button("🎲 Surprise Me!", key="btn_cbr_surprise"):
                st.session_state['cbr_surprise'] = surprise_me(animes)

    show_surprise_result(key='cbr_surprise')
    st.divider()

    all_titles = animes['title'].sort_values().tolist()
    selected_titles = st.multiselect(
        "Search and select anime you like:",
        options=all_titles,
        placeholder="e.g. Naruto, Bleach, One Piece..."
    )

    exclude = st.checkbox("Exclude sequels & related titles", value=True)

    if selected_titles:
        st.subheader("Your Selection")
        cols = st.columns(7)
        for i, title in enumerate(selected_titles):
            anime = animes[animes['title'] == title].iloc[0]
            with cols[i % 7]:
                show_anime_card(anime, key_suffix=f"sel_{i}")

        st.subheader("✨ Recommended For You")
        recs = get_cbr(selected_titles, animes, embeddings, exclude_franchise=exclude)
        cols = st.columns(7)
        for i, (_, anime) in enumerate(recs.head(14).iterrows()):
            with cols[i % 7]:
                show_anime_card(anime, key_suffix=f"rec_{i}")

        st.subheader("👥 Users Who Like This Also Like")
        assoc = give_associations(selected_titles, animes, rules, top_n=7)
        if not assoc.empty:
            cols = st.columns(7)
            for i, (_, anime) in enumerate(assoc.head(7).iterrows()):
                with cols[i % 7]:
                    show_anime_card(anime, key_suffix=f"assoc_{i}")
        else:
            st.info("No associations found for selected anime.")

elif page == 'search':
    left, right = st.columns([2, 1])
    with left:
        st.title("🔍 Search & Discover")
    with right:
        col1, col2 = st.columns([1, 1])
        with col1:
            st.write("")
            anime_search_bar(key="search_searchbar", show_label=False)
        with col2:
            st.write("")
            if st.button("🎲 Surprise Me!", key="btn_search_surprise"):
                st.session_state['surprise_search'] = surprise_me(animes)

    show_surprise_result(key='surprise_search')
    st.divider()

    st.subheader("🔎 Search by Description")
    query = st.text_input("Describe the kind of anime you're looking for:",
                          placeholder="e.g. a boy who wants to become the strongest fighter...")
    if query:
        results = search_by_synopsis(query, animes, embeddings, model, top_n= 7)
        cols = st.columns(7)
        for i, (_, anime) in enumerate(results.iterrows()):
            with cols[i % 7]:
                show_anime_card(anime, key_suffix=f"synopsis_{i}")

    st.divider()

    st.subheader("🎛️ Filter Anime")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        genre_filter = st.selectbox("Genre", options=['Any'] + sorted([
            'Action', 'Adventure', 'Comedy', 'Drama', 'Fantasy',
            'Horror', 'Romance', 'Sci-Fi', 'Thriller', 'Slice of Life'
        ]))
    with col2:
        min_score_filter = st.slider("Minimum Score", 0.0, 10.0, 7.0, step=0.1)
    with col3:
        max_episodes_filter = st.number_input("Max Episodes", min_value=1, max_value=2000, value=100)
    with col4:
        min_members_filter = st.number_input("Min Members", min_value=0, max_value=1000000, value=1000, step=1000)

    if st.button("🔍 Apply Filters"):
        st.session_state['filtered_results'] = filter_animes(
            animes,
            genre=None if genre_filter == 'Any' else genre_filter,
            max_episodes=max_episodes_filter,
            min_score=min_score_filter,
            min_members=min_members_filter
        )

    if 'filtered_results' in st.session_state and st.session_state['filtered_results'] is not None:
        filtered = st.session_state['filtered_results']
        st.write(f"Found **{len(filtered)}** anime")
        cols = st.columns(7)
        for i, (_, anime) in enumerate(filtered.head(14).iterrows()):
            with cols[i % 7]:
                show_anime_card(anime, key_suffix=f"filter_{i}")

        st.divider()


elif page == 'results':
    query = st.session_state.get('search_query', '')
    st.title(f"🔍 Results for: '{query}'")

    if st.button("← Back"):
        st.session_state['page'] = st.session_state.get('prev_page', 'home')
        st.rerun()

    results = animes[animes['title'].str.contains(query, case=False, na=False)]
    if not results.empty:
        st.write(f"Found **{len(results)}** anime")
        cols = st.columns(7)
        for i, (_, anime) in enumerate(results.iterrows()):
            with cols[i % 7]:
                show_anime_card(anime, key_suffix=f"res_{i}")
    else:
        st.info("No anime found.")

# ── Detail Page ───────────────────────────────────────────────────────────────
elif page == 'detail':
    uid = st.session_state.get('detail_uid')
    prev = st.session_state.get('prev_page', 'home')

    if st.button("← Back"):
        st.session_state['page'] = prev
        st.rerun()

    if uid is None:
        st.error("No anime selected.")
    else:
        # Look up by uid, fall back to title match
        row = animes[animes['uid'] == uid] if 'uid' in animes.columns else animes[animes['title'] == uid]

        if row.empty:
            st.error("Anime not found.")
        else:
            anime = row.iloc[0]

            # ── Header ────────────────────────────────────────────────────────
            col_img, col_info = st.columns([1, 3])

            with col_img:
                img_url = anime.get('img_url', '')
                if pd.isna(img_url) or img_url == '' or not is_valid_image(img_url):
                    st.image(PLACEHOLDER_IMG, width=220)
                else:
                    st.image(img_url, width=220)

            with col_info:
                st.title(anime['title'])

                score   = anime.get('score', 'N/A')
                members = anime.get('members', 'N/A')
                eps     = anime.get('episodes', 'N/A')
                genre   = anime.get('genre', 'N/A')
                anime_type = anime.get('type', 'N/A')
                status  = anime.get('status', 'N/A')
                aired   = anime.get('aired', 'N/A')
                rating  = anime.get('rating', 'N/A')

                st.markdown(f"⭐ **Score:** {score} &nbsp;|&nbsp; 👥 **Members:** {members:,}" 
                            if isinstance(members, (int, float)) else
                            f"⭐ **Score:** {score} &nbsp;|&nbsp; 👥 **Members:** {members}")
                st.markdown(f"🎬 **Episodes:** {eps} &nbsp;|&nbsp; 📺 **Type:** {anime_type}")
                st.markdown(f"🎭 **Genre:** {genre}")
                st.markdown(f"📅 **Aired:** {aired} &nbsp;|&nbsp; 🔖 **Status:** {status}")
                if rating != 'N/A':
                    st.markdown(f"🔞 **Rating:** {rating}")

            st.divider()

            # ── Synopsis ──────────────────────────────────────────────────────
            st.subheader("📖 Synopsis")
            synopsis = anime.get('synopsis', '')
            if pd.isna(synopsis) or synopsis == '':
                st.info("No synopsis available.")
            else:
                st.write(synopsis)

            st.divider()

            # ── Similar Anime ─────────────────────────────────────────────────
            st.subheader("✨ Similar Anime")
            try:
                recs = get_cbr([anime['title']], animes, embeddings, exclude_franchise=True)
                if not recs.empty:
                    cols = st.columns(7)
                    for i, (_, rec) in enumerate(recs.head(7).iterrows()):
                        with cols[i % 7]:
                            show_anime_card(rec, key_suffix=f"detail_rec_{i}")
                else:
                    st.info("No similar anime found.")
            except Exception as e:
                st.info("Could not load similar anime.")

            # ── Associations ──────────────────────────────────────────────────
            st.subheader("👥 Users Who Like This Also Like")
            try:
                assoc = give_associations([anime['title']], animes, rules, top_n=7)
                if not assoc.empty:
                    cols = st.columns(7)
                    for i, (_, rec) in enumerate(assoc.head(7).iterrows()):
                        with cols[i % 7]:
                            show_anime_card(rec, key_suffix=f"detail_assoc_{i}")
                else:
                    st.info("No associations found.")
            except Exception:
                st.info("Could not load associations.")