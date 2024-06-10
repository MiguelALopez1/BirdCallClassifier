import pandas as pd
from pathlib import Path

# Define paths
data_path = Path.cwd()/'BirdSongsEurope'
metadata_file = data_path/'metadata.csv'

# Read metadata file
df = pd.read_csv(metadata_file)

# Correct the relative paths for audio files
df['relative_path'] = 'mp3/' + df['Path'].apply(lambda x: x.split('/')[-1])  # Ensure this matches the actual file structure

# Select relevant columns
df = df[['relative_path', 'Species']]
df.to_csv(data_path/'prepared_metadata.csv', index=False)
print("Metadata prepared and saved to:", data_path/'prepared_metadata.csv')
