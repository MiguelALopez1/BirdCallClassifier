import torchaudio
from tqdm import tqdm
import pandas as pd

RATE_HZ = 16000
MAX_SECONDS = 10
MAX_LENGTH = RATE_HZ * MAX_SECONDS

def split_audio(file):
    try:
        audio, rate = torchaudio.load(str(file))
        num_segments = len(audio[0]) // MAX_LENGTH
        segmented_audio = []

        for i in range(num_segments):
            start = i * MAX_LENGTH
            end = min((i + 1) * MAX_LENGTH, len(audio[0]))
            segment = audio[0][start:end]
            transform = torchaudio.transforms.Resample(rate, RATE_HZ)
            segment = transform(segment).squeeze(0).numpy().reshape(-1)
            segmented_audio.append(segment)

        df_segments = pd.DataFrame({'audio': segmented_audio})
        return df_segments
    except Exception as e:
        print(f"Error processing file: {e}")
        return None

def preprocess_data(df):
    df_list = []
    for input_file, input_label in tqdm(zip(df['file'].values, df['label'].values)):
        resulting_df = split_audio(input_file)
        if resulting_df is not None:
            resulting_df['label'] = input_label
            df_list.append(resulting_df)

    df = pd.concat(df_list, axis=0)
    df = df[~df['audio'].isnull()]
    return df
