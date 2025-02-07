# Movies-Recommender

This project is a movie recommender system that uses both heuristic and model-based approaches to find similar movies. The system leverages movie embeddings and various movie attributes to compute similarities. An additional feature is provided which consists of a hybrid recommendation based on both movies similarity and user ratings.

## Crucial Elements Used

This project leverages several key techniques and models to generate movie recommendations:

- **Sentence-BERT (sentence-transformers/all-MiniLM-L6-v2)**: This pre-trained model is used to convert movie descriptions and attributes into dense vector embeddings. By leveraging Sentence-BERT, the project captures semantic relationships between movie descriptions, which helps in generating more accurate and context-aware recommendations.
  
- **SVD (Singular Value Decomposition)**: SVD is employed for movie recommendation based on the users ratings. This technique helps in extracting the most significant features from high-dimensional embeddings, reducing noise, and improving the accuracy and efficiency of similarity searches.
  
- **Pairwise Cosine Similarity**: To measure the similarity between movies, the project uses pairwise cosine similarity, which calculates the cosine of the angle between two movie embeddings. This metric is particularly effective for determining how similar two items are based on their vector representations, making it ideal for content-based recommendation systems.



## Example

In the Jupyter notebook, you can compare the results of heuristic and model-based similarity searches:

```python3
import pandas as pd
import numpy as np
import json
import ast
from utils import model_similarity, heuristic_similarity, find_movie

# Load the dataset
df_movies = pd.read_csv('dataset/movies_data_embeddings.csv', low_memory=False)
df_movies['embeddings'] = df_movies['embeddings'].apply(json.loads)
df_movies['embeddings'] = df_movies['embeddings'].apply(np.array)
df_movies['genres'] = df_movies['genres'].apply(ast.literal_eval)
df_movies['tags'] = df_movies['tags'].apply(ast.literal_eval)
df_movies['overview_keywords'] = df_movies['overview'].apply(str)
df_movies['characters'] = df_movies['characters'].apply(ast.literal_eval)
df_movies['directors'] = df_movies['directors'].apply(ast.literal_eval)
df_movies['actors'] = df_movies['actors'].apply(ast.literal_eval)

# Find similar movies using heuristic similarity
heuristic_results = heuristic_similarity(df_movies, 'The Dark Knight', 10)["title"]

# Find similar movies using model similarity
model_results = model_similarity(df_movies, 'The Dark Knight', 10)["title"]
```



## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgements

- The dataset used in this project is sourced from [Kaggle](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset).
- The project uses `scikit-learn` for computing cosine similarity and `pandas` for data manipulation.
- The **Sentence-BERT model** used in this project is [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2), which is available on Hugging Face. This model is used for generating high-quality sentence embeddings that capture semantic information, making it ideal for the recommendation system's content-based similarity approach.

Results of my experiments may be viewed the `compare_model_and_heuristic_search.ipynb` notebook. For more details, refer to the code and comments in the Jupyter notebook and `utils.py`.
