from pathlib import Path
import pandas as pd

# path to dataset
download_path = Path.cwd() / 'BirdSongsEurope'

# read metadata file as a data frame
metadata_file = download_path / 'metadata.csv'
df = pd.read_csv(metadata_file)

# construct file path by concatenating folder and file name
df['relative_path'] = df['Path'].astype(str)

# Take relevant columns
df = df[['relative_path', 'Species']]

# Save the modified metadata
prepared_metadata_file = download_path / 'prepared_metadata.csv'
df.to_csv(prepared_metadata_file, index=False)

print(f"Metadata prepared and saved to: {prepared_metadata_file}")
