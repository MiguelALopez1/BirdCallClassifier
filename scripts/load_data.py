import pandas as pd
from pathlib import Path

def load_data(data_path):
    file_list = []
    label_list = []
    for file in Path(data_path).glob('*.mp3'):
        label = file.stem.split('-')[0]
        file_list.append(str(file))
        label_list.append(label)

    df = pd.DataFrame({'file': file_list, 'label': label_list})
    return df
