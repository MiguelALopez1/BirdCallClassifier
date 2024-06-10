
import torch
from torch.utils.data import DataLoader, Dataset, random_split
import torchaudio
from pathlib import Path
import pandas as pd
import random

class BirdSongDS(Dataset):
    def __init__(self, df, data_path):
        self.df = df
        self.data_path = data_path
        self.duration = 4000
        self.sr = 44100
        self.channel = 2
        self.shift_pct = 0.4

        # Create a mapping from species names to numerical labels
        self.species_to_label = {species: idx for idx, species in enumerate(df['Species'].unique())}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        audio_file = self.data_path / self.df.loc[idx, 'relative_path']
        if not audio_file.is_file():
            print(f"File {audio_file} does not exist. Skipping.")
            return self.__getitem__((idx + 1) % len(self))

        species_name = self.df.loc[idx, 'Species']
        label = self.species_to_label[species_name]

        aud = torchaudio.load(audio_file)
        reaud = self.resample(aud, self.sr)
        rechan = self.rechannel(reaud, self.channel)
        dur_aud = self.pad_trunc(rechan, self.duration)
        shift_aud = self.time_shift(dur_aud, self.shift_pct)
        sgram = self.spectro_gram(shift_aud, n_mels=64, n_fft=1024, hop_len=None)
        aug_sgram = self.spectro_augment(sgram, max_mask_pct=0.1, n_freq_masks=2, n_time_masks=2)

        return torch.tensor(aug_sgram, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

    @staticmethod
    def resample(aud, newsr):
        sig, sr = aud
        if sr == newsr:
            return aud
        num_channels = sig.shape[0]
        resig = torchaudio.transforms.Resample(sr, newsr)(sig[:1, :])
        if num_channels > 1:
            retwo = torchaudio.transforms.Resample(sr, newsr)(sig[1:, :])
            resig = torch.cat([resig, retwo])
        return resig, newsr

    @staticmethod
    def rechannel(aud, new_channel):
        sig, sr = aud
        if sig.shape[0] == new_channel:
            return aud
        if new_channel == 1:
            resig = sig[:1, :]
        else:
            resig = torch.cat([sig, sig])
        return resig, sr

    @staticmethod
    def pad_trunc(aud, max_ms):
        sig, sr = aud
        num_rows, sig_len = sig.shape
        max_len = sr // 1000 * max_ms
        if sig_len > max_len:
            sig = sig[:, :max_len]
        elif sig_len < max_len:
            pad_begin_len = random.randint(0, max_len - sig_len)
            pad_end_len = max_len - sig_len - pad_begin_len
            pad_begin = torch.zeros((num_rows, pad_begin_len))
            pad_end = torch.zeros((num_rows, pad_end_len))
            sig = torch.cat((pad_begin, sig, pad_end), 1)
        return sig, sr

    @staticmethod
    def time_shift(aud, shift_limit):
        sig, sr = aud
        _, sig_len = sig.shape
        shift_amt = int(random.random() * shift_limit * sig_len)
        return sig.roll(shift_amt), sr

    @staticmethod
    def spectro_gram(aud, n_mels=64, n_fft=1024, hop_len=None):
        sig, sr = aud
        top_db = 80
        spec = torchaudio.transforms.MelSpectrogram(sr, n_fft=n_fft, hop_length=hop_len, n_mels=n_mels)(sig)
        spec = torchaudio.transforms.AmplitudeToDB(top_db=top_db)(spec)
        return spec

    @staticmethod
    def spectro_augment(spec, max_mask_pct=0.1, n_freq_masks=1, n_time_masks=1):
        _, n_mels, n_steps = spec.shape
        mask_value = spec.mean()
        aug_spec = spec
        freq_mask_param = max_mask_pct * n_mels
        for _ in range(n_freq_masks):
            aug_spec = torchaudio.transforms.FrequencyMasking(freq_mask_param)(aug_spec, mask_value)
        time_mask_param = max_mask_pct * n_steps
        for _ in range(n_time_masks):
            aug_spec = torchaudio.transforms.TimeMasking(time_mask_param)(aug_spec, mask_value)
        return aug_spec

# Load the metadata
data_path = Path.cwd() / 'BirdSongsEurope'
metadata_file = data_path / 'prepared_metadata.csv'
df = pd.read_csv(metadata_file)
print("Initial data path:", data_path)
print("First few relative paths in metadata:")
print(df['relative_path'].head())

# Create dataset and dataloaders
dataset = BirdSongDS(df, data_path)
num_items = len(dataset)
num_train = round(num_items * 0.8)
num_val = num_items - num_train
train_ds, val_ds = random_split(dataset, [num_train, num_val])

train_dl = DataLoader(train_ds, batch_size=16, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=16, shuffle=False)

# Save the dataloaders
with open('train_dl.pth', 'wb') as f:
    torch.save(train_dl, f)
with open('val_dl.pth', 'wb') as f:
    torch.save(val_dl, f)
