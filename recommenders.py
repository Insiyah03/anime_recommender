import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Baysian Scores
def get_top_animes(animes, top_n):
    return  (
        animes.sort_values('weighted_score', ascending=False)
        .drop_duplicates(subset='base_title', keep='first')
        .head(top_n)
    )

def get_most_popular_animes(animes,top_n):
    return (
        animes.sort_values('popularity')
        .drop_duplicates(subset='base_title',keep='first')
        .head(top_n)
    )


def get_top_genres(animes,genre,top_n):
    return(
        animes[animes['genre'].str.contains(genre, case=False, na=False)]
        .sort_values('weighted_score', ascending=False)
        .drop_duplicates(subset='base_title', keep='first')
        .head(top_n)
    )

# Content based
def get_cbr(titles, animes, embeddings, top_n=20, exclude_franchise=True):
    if isinstance(titles, str):
        titles = [titles]

    # use positional indices not pandas index
    indices = animes[animes['title'].isin(titles)].index.tolist()
    positions = [animes.index.get_loc(idx) for idx in indices]

    query_embedding = embeddings[positions].mean(axis=0).reshape(1, -1)

    similarities = cosine_similarity(query_embedding, embeddings)[0]
    similar_indices = similarities.argsort()[::-1]

    recs = animes.iloc[similar_indices]

    if exclude_franchise:
        # get the base titles of selected anime
        base_titles = animes.iloc[positions]['base_title'].tolist()
        
        # exclude if any base title is contained in the rec's title
        def is_related(title):
            return any(base.lower() in title.lower() for base in base_titles)
        
        recs = recs[~recs['title'].apply(is_related)]

    

    return recs.head(top_n)



def search_by_synopsis(query, animes, embeddings, model, top_n=5):
    # embed the user's query
    query_embedding = model.encode([query])
    
    # compute similarity against all anime embeddings
    similarities = cosine_similarity(query_embedding, embeddings)[0]
    
    # get top N most similar
    top_indices = similarities.argsort()[::-1][:top_n]
    
    recs = animes.iloc[top_indices].copy()
    recs['similarity'] = similarities[top_indices]
    
    return recs

# Associations
def give_associations(titles, animes, rules, top_n=10):
    if isinstance(titles, str):
        titles = [titles]

    results = []
    for title in titles:
        uid = animes[animes['title'] == title].iloc[0]['uid']
        selected = frozenset([str(uid)])
        recs = rules[rules['antecedents'] == selected]

        for r in recs.consequents:
            aid = int(list(r)[0])
            match = animes[animes['uid'] == aid]
            if not match.empty:
                results.append(match.iloc[0])

    if not results:
        return pd.DataFrame()

    return (
        pd.DataFrame(results)
        .drop_duplicates(subset='title')
        .head(top_n)
    )


# Extra
def filter_animes(animes, genre=None, max_episodes=None, min_score=None, min_members=None):
    df = animes.copy()
    
    if genre:
        df = df[df['genre'].str.contains(genre, case=False, na=False)]
    if max_episodes:
        df = df[df['episodes'] <= max_episodes]
    if min_score:
        df = df[df['score'] >= min_score]
    if min_members:
        df = df[df['members'] >= min_members]
    
    return df

def surprise_me(animes, min_score=8.0):
    df = animes[animes['score'] >= min_score]
    if df.empty:
        return None
    return df.sample(1).iloc[0]