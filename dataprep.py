import pandas as pd

# ===[ 1. Load Data from Google Drive ]===
base_path = "/content/drive/MyDrive/DMS672_Course_Project/"

# Load title.basics.tsv.gz (only movies)
chunks = pd.read_csv(base_path + "title.basics.tsv.gz", sep='\t', compression='gzip', na_values='\\N', chunksize=100000)
df_basics = pd.concat(chunk[chunk['titleType'] == 'movie'] for chunk in chunks)
df_basics = df_basics[['tconst', 'primaryTitle', 'originalTitle', 'startYear', 'runtimeMinutes', 'genres']].dropna()
df_basics['year'] = df_basics['startYear'].astype(int)
df_basics[['genre1', 'genre2', 'genre3']] = df_basics['genres'].str.split(',', expand=True)

# Load title.ratings.tsv.gz
df_ratings = pd.read_csv(base_path + "title.ratings.tsv.gz", sep='\t', compression='gzip', na_values='\\N')

# Load title.crew.tsv.gz
df_crew = pd.read_csv(base_path + "title.crew.tsv.gz", sep='\t', compression='gzip', na_values='\\N')

# Load name.basics.tsv.gz (for mapping names)
df_names = pd.read_csv(base_path + "name.basics.tsv.gz", sep='\t', compression='gzip', na_values='\\N')
name_map = dict(zip(df_names['nconst'], df_names['primaryName']))

# Map directorName
df_crew['directorName'] = df_crew['directors'].apply(lambda x: name_map.get(x, '') if pd.notna(x) else '')

# FIXED: Map writerNames safely
def map_writer_ids(writer_str):
    if pd.isna(writer_str):
        return ''
    ids = writer_str.split(',')
    names = []
    for i in ids:
        name = name_map.get(i, '')
        if pd.notna(name):
            names.append(str(name))
    return ', '.join(names)

df_crew['writerNames'] = df_crew['writers'].apply(map_writer_ids)

# Load title.principals.tsv.gz (only actors/actresses)
chunks = pd.read_csv(base_path + "title.principals.tsv.gz", sep='\t', compression='gzip', na_values='\\N', chunksize=100000)
df_principals = pd.concat(chunk[chunk['category'].isin(['actor', 'actress'])] for chunk in chunks)

# Get top 3 actors per movie
df_top_actors = df_principals.sort_values(by=['tconst', 'ordering']).groupby('tconst')['nconst'].apply(lambda x: list(x[:3])).reset_index()
df_top_actors.columns = ['tconst', 'topActorIDs']

# FIXED: Map actor names safely
def get_actor_names(nconst_list):
    names = []
    for i in nconst_list:
        name = name_map.get(i, '')
        if pd.notna(name):
            names.append(str(name))
    return names

df_top_actors['actorNames'] = df_top_actors['topActorIDs'].apply(get_actor_names)

# ===[ 2. Merge All DataFrames ]===
df_final = df_basics.merge(df_ratings, on='tconst', how='inner')
df_final = df_final.merge(df_crew[['tconst', 'directors', 'directorName', 'writers', 'writerNames']], on='tconst', how='left')
df_final = df_final.merge(df_top_actors, on='tconst', how='left')

# ===[ 3. Clean and Select Final Columns ]===
df_final = df_final[[
    'tconst',
    'primaryTitle', 'originalTitle',
    'year', 'runtimeMinutes',
    'genres', 'genre1', 'genre2', 'genre3',
    'averageRating', 'numVotes',
    'directors', 'directorName',
    'writers', 'writerNames',
    'topActorIDs', 'actorNames'
]]

df_final = df_final.dropna(subset=['primaryTitle', 'genres', 'averageRating', 'actorNames'])
df_final = df_final.reset_index(drop=True)

# ===[ 4. Save and Download as CSV ]===
df_final.to_csv('final_imdb_dataset.csv', index=False)

from google.colab import files
files.download('final_imdb_dataset.csv')
