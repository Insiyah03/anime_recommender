# 🍜 Anime Recommendation System

> 🚀 **Live Demo:** [_Add deployment link here_]

---

## 📌 Project Overview

A full-stack anime recommendation web application that helps users discover anime they'll love through multiple intelligent recommendation strategies. The system addresses the **cold-start problem** common in recommendation engines by combining content-based filtering, association rule mining, Bayesian scoring, and semantic search — all wrapped in an interactive Streamlit UI.

**Key outcomes:**
- Users can find similar anime based on titles they already enjoy
- A semantic search engine lets users describe what they want in natural language
- Advanced filters (genre, score, episodes, popularity) give fine-grained control
- A "Surprise Me" feature surfaces hidden gems above a quality threshold

---

## 📂 Dataset

> 📎 **Dataset:** [_Add dataset link here (e.g., Kaggle)_]

The dataset contains three components:

| File | Description |
|---|---|
| `animes.csv` | Metadata for ~14,900 anime titles including title, genre, score, members, synopsis, episode count, type, status, and cover image URLs |
| `reviews.csv` | User-written reviews with structured sub-scores across story, animation, sound, character, and enjoyment |
| `profiles.csv` | User profiles with lists of favourite anime used to mine co-occurrence patterns |

The reviews data informed the choice of recommendation strategy (no item-based CF due to one review per user), the profiles data powered association rule mining, and the anime metadata drove content-based and knowledge-based filtering.

---

## ✨ Features

- **🏆 Top Rated & 🔥 Most Popular** — Bayesian-weighted ranking that balances score quality with audience size, preventing low-vote anime from dominating
- **🎯 Content-Based Recommendations** — Multi-title selection with averaged embeddings; excludes franchise sequels intelligently using base-title parsing
- **👥 "Users Also Like"** — Association rules mined from user favourite lists using FP-Growth, surfacing real community co-viewing patterns
- **🔎 Semantic Search** — Natural language synopsis search powered by `all-mpnet-base-v2` sentence embeddings and cosine similarity
- **🎛️ Advanced Filters** — Filter by genre, minimum score, maximum episodes, and minimum member count
- **🎲 Surprise Me** — Random recommendation from high-quality anime (score ≥ 8.0)


---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| **Language** | Python 3.x |
| **Web Framework** | Streamlit |
| **ML / NLP** | Sentence Transformers (`all-mpnet-base-v2`), scikit-learn |
| **Association Mining** | mlxtend (FP-Growth, Association Rules) |
| **Data Processing** | pandas, NumPy, ast, re |
| **Similarity** | Cosine Similarity (sklearn) |
| **Serialization** | Parquet, NumPy `.npy`, Pickle |
| **Frontend** | Streamlit Components (custom HTML/CSS/JS carousels) |
| **HTTP** | requests (image validation) |

---

## ⚙️ Installation & Setup

### Prerequisites

- Python 3.8+
- pip

### 1. Clone the repository

```bash
git clone https://github.com/your-username/anime-recommender.git
cd anime-recommender
```

### 2. Install dependencies

```bash
pip install streamlit pandas numpy scikit-learn sentence-transformers mlxtend requests pyarrow
```

### 3. Prepare the data

Place the raw CSV files in `Raw_Data/`:
```
Raw_Data/
├── animes.csv
├── reviews.csv
└── profiles.csv
```

Then run the preprocessing notebooks in order:
```bash
# 1. Data preprocessing (animes, reviews, profiles)
jupyter notebook notebooks/preprocessing.ipynb

# 2. Generate synopsis embeddings
jupyter notebook notebooks/content_based.ipynb

# 3. Mine association rules
jupyter notebook notebooks/associations.ipynb
```

This will populate:
```
data/
├── animes.parquet
├── reviews.parquet
└── profiles.parquet

models/
├── embeddings.npy
└── rules.pkl
```

### 4. Run the app

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`.

---

## 🚀 Usage

| Page | How to Use |
|---|---|
| **Home** | Browse top-rated and popular anime via carousels; filter by genre |
| **Content Based** | Select one or more anime you like → get similar recommendations + community associations |
| **Search** | Describe an anime in plain English, or use filters to narrow results |
| **Detail View** | Click any anime card to see full metadata, synopsis, similar titles, and community picks |
| **Surprise Me** | Click 🎲 on any page for a random quality recommendation |

---

## 🗂️ Project Structure

```
anime-recommender/
│
├── app.py                  # Main Streamlit application & UI routing
├── recommenders.py         # All recommendation logic (CBR, associations, filters, etc.)
│
├── notebooks/
│   ├── preprocessing.ipynb # Data cleaning & EDA for animes, reviews, profiles
│   ├── content_based.ipynb # Sentence Transformer embedding generation
│   └── associations.ipynb  # FP-Growth association rule mining
│
├── data/                   # Processed Parquet files (generated)
├── models/                 # Saved embeddings and rules (generated)
└── Raw_Data/               # Original CSV source files
```

---

## 💡 Key Highlights & What I Learned

**EDA-Driven Architecture Decisions**
Through exploratory analysis, I discovered that users in this dataset had only written one review each, making traditional user-based collaborative filtering impossible. This insight directly shaped the system design: pivoting to Bayesian scoring, content-based filtering, and association rule mining from favourites lists rather than ratings.

**Bayesian Scoring**
Implemented a weighted rating formula (analogous to IMDb's formula) using the global mean score and the 75th percentile members count as a confidence threshold. This prevents niche anime with a handful of perfect scores from outranking well-loved titles.

**Semantic Embeddings for Cold Start**
Used `all-mpnet-base-v2` (512-token max sequence length, strong semantic performance) to encode all anime synopses into dense vectors. These embeddings support both content-based recommendations and free-text search — meaning users don't need to know an exact title to find what they want.

**Multi-Title Embedding Fusion**
When a user selects multiple anime, their embeddings are averaged into a single query vector. This elegantly handles blended taste profiles without requiring user history.

**FP-Growth Association Rules**
Mined co-occurrence patterns from user favourite lists using FP-Growth with `min_support=0.01` and `min_lift=1.2`. This surfaces community-driven associations that pure content similarity misses (e.g., genre-crossing recommendations that fans actually watch together).


---

## 🔮 Future Improvements

- **Hybrid Recommender** — Combine content-based and association scores into a unified ranking with tunable weights
- **Sentiment-Aware Scoring** — Use the cleaned review text for sentiment analysis to augment raw score signals
- **Trailer Integration** — Embed YouTube trailers on detail pages via the MyAnimeList or Jikan API
- **Containerisation** — Dockerise the app for easier deployment and reproducibility


---

## 📄 License

This project is for educational and portfolio purposes.