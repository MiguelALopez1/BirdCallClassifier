from pathlib import Path
import torch
import pandas as pd
from torch.utils.data import DataLoader, random_split
from dataset import SoundDS

# Path to the prepared metadata file
prepared_metadata_file = Path.cwd()/'UrbanSound8K'/'metadata'/'prepared_metadata.csv'

# Load the prepared metadata
df = pd.read_csv(prepared_metadata_file)

# Path to the audio data
data_path = Path.cwd()/'UrbanSound8K'

# Create the dataset
myds = SoundDS(df, data_path)

# Random split of 80:20 between training and validation
num_items = len(myds)
num_train = round(num_items * 0.8)
num_val = num_items - num_train
train_ds, val_ds = random_split(myds, [num_train, num_val])

# Create training and validation data loaders
train_dl = DataLoader(train_ds, batch_size=16, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=16, shuffle=False)

# Save the dataloaders for use in the training script
torch.save(train_dl, 'train_dl.pth')
torch.save(val_dl, 'val_dl.pth')

# Check the dataloaders
for data in train_dl:
    inputs, labels = data
    print(inputs.shape, labels.shape)
    break

for data in val_dl:
    inputs, labels = data
    print(inputs.shape, labels.shape)
    break
