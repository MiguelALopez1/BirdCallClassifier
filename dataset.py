from torch.utils.data import DataLoader, Dataset, random_split
import torchaudio
import torch
from audio_util import AudioUtil
import pandas as pd
from pathlib import Path

class BirdSongDS(Dataset):
    def __init__(self, df, data_path):
        self.data_path = str(data_path)
        self.duration = 4000
        self.sr = 44100
        self.channel = 2
        self.shift_pct = 0.4

        # Create a mapping from species names to integer labels
        self.species_to_label = {species: i for i, species in enumerate(df['Species'].unique())}
        
        # Map species names to labels
        df['label'] = df['Species'].map(self.species_to_label)

        # Print initial paths for debugging
        print(f"Initial data path: {self.data_path}")
        print("First few relative paths in metadata:")
        print(df['relative_path'].head())

        # Validate paths and filter out missing files
        def check_file_exists(x):
            file_path = Path(self.data_path + '/' + x)
            exists = file_path.is_file()
            if not exists:
                print(f"File {file_path} does not exist.")
            return exists

        valid_paths = df['relative_path'].apply(check_file_exists)
        self.df = df[valid_paths]
        missing_files_count = len(df) - len(self.df)
        if missing_files_count > 0:
            print(f"Filtered out {missing_files_count} missing files.")
        self.df.reset_index(drop=True, inplace=True)

    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        audio_file = self.data_path + '/' + self.df.loc[idx, 'relative_path']
        label = self.df.loc[idx, 'label']
        
        aud = AudioUtil.open(audio_file)
        reaud = AudioUtil.resample(aud, self.sr)
        rechan = AudioUtil.rechannel(reaud, self.channel)
        dur_aud = AudioUtil.pad_trunc(rechan, self.duration)
        shift_aud = AudioUtil.time_shift(dur_aud, self.shift_pct)
        sgram = AudioUtil.spectro_gram(shift_aud, n_mels=64, n_fft=1024, hop_len=None)
        aug_sgram = AudioUtil.spectro_augment(sgram, max_mask_pct=0.1, n_freq_masks=2, n_time_masks=2)

        return aug_sgram.clone().detach().to(torch.float32), torch.tensor(label, dtype=torch.long)

if __name__ == "__main__":
    prepared_metadata_file = Path.cwd()/'BirdSongsEurope'/'prepared_metadata.csv'
    df = pd.read_csv(prepared_metadata_file)
    data_path = Path.cwd()/'BirdSongsEurope'
    myds = BirdSongDS(df, data_path)

    # Random split of 80:20 between training and validation
    num_items = len(myds)
    num_train = round(num_items * 0.8)
    num_val = num_items - num_train
    train_ds, val_ds = random_split(myds, [num_train, num_val])

    # Create training and validation data loaders
    train_dl = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=16, shuffle=False)

    torch.save(train_dl, 'train_dl.pth')
    torch.save(val_dl, 'val_dl.pth')
