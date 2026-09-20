# IMDb Content-Based Movie Recommender

A content-based movie recommendation system built on IMDb's public datasets. The project cleans and merges raw IMDb data, explores it, engineers features, and builds a hybrid recommender that combines TF-IDF text similarity (genres, director, cast) with numerical similarity (year, runtime, weighted rating) to suggest movies similar to a given title.

## Overview

Given a movie title and its release year, the recommender returns a ranked list of similar movies based on:

- **Content similarity** — genres, director, and top-billed actors, combined into a weighted "soup" and compared using TF-IDF + cosine similarity.
- **Numerical similarity** — closeness in release year, runtime, and IMDb weighted rating, with weights that adapt depending on runtime (e.g. short films are compared differently from feature-length films) and how far apart the years are.

The two similarity scores are blended (60% content, 40% numerical closeness) to produce a final ranked recommendation list.

## Project Structure

```
.
├── dataprep.py            # Downloads & merges raw IMDb .tsv.gz files into one CSV
├── movie_rec_eda.ipynb    # Exploratory data analysis on the merged dataset
├── movie_recfinal.ipynb   # Cleaning, feature engineering, and the recommender itself
└── README.md
```

## Data Source

The pipeline is built on IMDb's publicly available non-commercial datasets:
[https://datasets.imdbws.com/](https://datasets.imdbws.com/)

Files used:

| File | Purpose |
|---|---|
| `title.basics.tsv.gz` | Titles, year, runtime, genres (filtered to `titleType == movie`) |
| `title.ratings.tsv.gz` | Average rating and number of votes per title |
| `title.crew.tsv.gz` | Director and writer IDs |
| `name.basics.tsv.gz` | Maps person IDs (`nconst`) to real names |
| `title.principals.tsv.gz` | Cast/crew credits (filtered to top 3 actors/actresses per title) |

## Pipeline

### 1. Data Preparation — `dataprep.py`

Run in an environment with the raw IMDb files available (the script assumes a Google Colab + Google Drive setup).

- Loads `title.basics` in chunks and keeps only movies.
- Splits `genres` into `genre1`, `genre2`, `genre3`.
- Loads `title.ratings`, `title.crew`, and `name.basics`.
- Maps director/writer IDs to real names via `name.basics`.
- Extracts the top 3 billed actors per title from `title.principals` and maps their IDs to names.
- Merges everything into a single dataframe on `tconst` and drops rows missing a title, genre, rating, or cast.
- Exports the result as `final_imdb_dataset.csv`.

**To run:**

```bash
python dataprep.py
```

> Update `base_path` in the script to point at your local copy of the raw `.tsv.gz` files if not running on Colab.

### 2. Exploratory Data Analysis — `movie_rec_eda.ipynb`

Takes `final_imdb_dataset.csv` as input and explores it, including:

- Missing value and duplicate analysis
- Genre distribution (primary/secondary/tertiary)
- Top-rated movies, directors, writers, and actors (with a minimum vote/movie-count threshold)
- IMDb weighted rating calculation (Bayesian-style, similar to IMDb's own "Top 250" formula)
- Runtime and rating distributions, outlier detection (IQR method)
- Trends by decade (volume, runtime, votes, genre popularity)
- Correlation matrix of numerical features
- Rating vs. vote-count relationships

### 3. Recommender — `movie_recfinal.ipynb`

Repeats the core cleaning steps from the EDA notebook, then:

1. **Feature engineering**
   - Converts stringified lists (genres, director, actor, writer names) back into Python lists.
   - Extracts `decade` and `decade_label`.
   - Min-max scales `runtimeMinutes` and `year`.

2. **Weighted "soup" construction** (`create_weighted_soup`)
   A single text string per movie built from:
   - Primary/secondary genre (weighted 12x / 8x — tertiary genre excluded)
   - Director (weighted 4x)
   - Top 3 actors (weighted 7x / 7x / 2x)

   This soup is vectorized with a TF-IDF vectorizer (`TfidfVectorizer(stop_words='english')`).

3. **Numerical distance** (`calculate_distance`)
   Combines scaled differences in year, weighted rating, and runtime. Weighting adapts based on:
   - Whether the reference movie is a short film (≤ 60 min)
   - How many years apart the two movies are (≤5, ≤15, or >15 years)

4. **Recommendation function** (`get_enhanced_recommendations(title, year, top_n=100)`)
   - Looks up the movie by (lowercased) title and year.
   - Computes cosine similarity between soups and numerical distance to every other movie.
   - Combines them: `combined_score = 0.6 * cosine_similarity + 0.4 * (1 - distance)`.
   - Returns the top-N most similar movies as a dataframe with director, cast, genres, rating, runtime, and similarity scores.

**Example usage:**

```python
df_recommended = get_enhanced_recommendations("Singham Returns", 2014, top_n=10)
df_recommended
```

## Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
```

Install with:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Notes

- Both notebooks were originally written for Google Colab (`/content/...` paths, `!pip install`, `google.colab.files.download`). Update file paths accordingly if running locally or on another platform.
- The recommendation function recomputes the TF-IDF matrix on every call; for repeated queries against the same dataset, consider caching `term_freq_matrix` and `cleaned_df['soup']` outside the function.

## License

Add a license of your choice (e.g. MIT) here. IMDb's datasets are subject to their own [non-commercial usage terms](https://developer.imdb.com/non-commercial-datasets/).
