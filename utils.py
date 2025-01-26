import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def find_movie(df, identifier):
    if isinstance(identifier, int):  
        result = df[df['id'] == identifier]
        if result.shape[0] == 0:
            print(f"No film found with id {identifier}.")
            return None
        return result

    elif isinstance(identifier, str):  
        result_exact = df[df['title'] == identifier]
        if result_exact.shape[0] == 1:
            return result_exact
        if result_exact.shape[0] > 1:
            print("Warning: Multiple films with the exact title. Please provide a unique title or id.")
            print("Found exact matches:")
            print(result_exact[['id', 'title']])  
            return None
        else:
            result_regex = df[df['title'].str.contains(identifier, flags=re.IGNORECASE, regex=True)]
            if result_regex.shape[0] > 1:
                print("Warning: Multiple films match the title pattern. Please provide a unique title or id.")
                print("Found matching titles:")
                print(result_regex[['id', 'title']])  
                return None
            if result_regex.shape[0] == 0:
                print(f"No films found with the title matching '{identifier}'.")
                return None
            return result_regex
    else:
        raise ValueError("Identifier must be either an integer (id) or a string (title)")
    
def get_content_similarity(df, movie_id, top_n=10):
    movie_embedding = df.loc[df['id'] == movie_id, 'embeddings'].values[0]
    movie_embedding = np.array(movie_embedding).reshape(1, -1)
    all_embeddings = np.vstack(df['embeddings'])
    similarities = cosine_similarity(movie_embedding, all_embeddings).flatten()
    df['similarity'] = similarities
    top_similar_movies = df.sort_values(by='similarity', ascending=False).head(top_n + 1)  
    return top_similar_movies.iloc[1:]  

def model_similarity(df, identifier, n=5):
    movie_1 = find_movie(df, identifier)
    if movie_1 is None:
        return None
    movie_1 = movie_1.squeeze()
    print(f"search: {movie_1['title']}")
    query_embedding = movie_1['embeddings']
    all_embeddings = list(df['embeddings'])
    similarities = cosine_similarity([query_embedding], all_embeddings)
    similar_indices = similarities.argsort()[0][::-1][:(n+1)]
    similar_indices = similar_indices[1:]
    return df.iloc[similar_indices][['id', 'title', 'genres', 'overview_keywords', 'tags', 'characters', 'directors', 'actors']]

def compare_multi_field(f1, f2):
    try:
        return len(set(f1).intersection(set(f2))) / len(set(f1).union(set(f2)))
    except:
        return 0

def heuristic_similarity(df, identifier, n=5, w_title = 2.2, w_adult = 0.1, w_original_language = 0.1, w_release_year = 0.1, w_genres = 0.9, w_overview_keyword = 0.4, w_popularity = 0.0, w_tags = 1.75, w_characters = 3.75, w_directors = 0.7, w_actors = 0.2):    
    movie_1 = find_movie(df, identifier)
    if movie_1 is None:
        return None
    movie_1 = movie_1.squeeze()
    print(f"search: {movie_1['title']}")
    movie_id = movie_1['id'].item()
    similarities = []
    for _, movie_2 in df.iterrows():
        if movie_2['id'] == movie_id:
            continue
        title = compare_multi_field(movie_1['title'].split(), [movie_2['title'].split()])
        adult = 1 if movie_1['adult'] == movie_2['adult'] else 0
        original_language = 1 if movie_1['original_language'] == movie_2['original_language'] else 0
        genres = compare_multi_field(movie_1['genres'], movie_2['genres'])
        overview_keywords = compare_multi_field(movie_1['overview_keywords'], movie_2['overview_keywords'])
        tags = compare_multi_field(movie_1['tags'], movie_2['tags'])
        characters = compare_multi_field(movie_1['characters'], movie_2['characters'])
        directors = compare_multi_field(movie_1['directors'], movie_2['directors'])
        actors = compare_multi_field(movie_1['actors'], movie_2['actors'])
        try:
            popularity = 1 - (abs(float(movie_1['popularity']) - float(movie_2['popularity'])) / 100)
        except:
            print(movie_1['popularity'], movie_2['popularity'], "!!!!!!!!!!!!!!!")
            popularity = 0
        similarity = w_title * title + w_adult * adult +  w_original_language * original_language + w_genres * genres + w_overview_keyword * overview_keywords + w_popularity * popularity + w_tags * tags + w_characters * characters + w_directors * directors + w_actors * actors
        similarities.append((movie_2['id'], similarity))
    similarities = sorted(similarities, key=lambda x: x[1], reverse=True)
    similar_indices = [x[0] for x in similarities[:n]]
    filtered_df = df[df['id'].isin(similar_indices)].reset_index(drop=True)
    filtered_df = filtered_df.set_index('id').loc[similar_indices].reset_index()
    return filtered_df[['id', 'title', 'genres', 'overview_keywords', 'tags', 'characters', 'directors', 'actors']]