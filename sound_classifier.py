import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd
from pathlib import Path
from dataset import BirdSongDS  # Ensure BirdSongDS is imported

class AudioClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        conv_layers = []
        self.conv1 = nn.Conv2d(2, 8, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(8)
        init.kaiming_normal_(self.conv1.weight, a=0.1)
        self.conv1.bias.data.zero_()
        conv_layers += [self.conv1, self.relu1, self.bn1]

        self.conv2 = nn.Conv2d(8, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(16)
        init.kaiming_normal_(self.conv2.weight, a=0.1)
        self.conv2.bias.data.zero_()
        conv_layers += [self.conv2, self.relu2, self.bn2]

        self.conv3 = nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.relu3 = nn.ReLU()
        self.bn3 = nn.BatchNorm2d(32)
        init.kaiming_normal_(self.conv3.weight, a=0.1)
        self.conv3.bias.data.zero_()
        conv_layers += [self.conv3, self.relu3, self.bn3]

        self.conv4 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.relu4 = nn.ReLU()
        self.bn4 = nn.BatchNorm2d(64)
        init.kaiming_normal_(self.conv4.weight, a=0.1)
        self.conv4.bias.data.zero_()
        conv_layers += [self.conv4, self.relu4, self.bn4]

        self.ap = nn.AdaptiveAvgPool2d(output_size=1)
        self.lin = nn.Linear(in_features=64, out_features=50)  # Adjust output size to number of species
        self.conv = nn.Sequential(*conv_layers)

    def forward(self, x):
        x = self.conv(x)
        x = self.ap(x)
        x = x.view(x.shape[0], -1)
        x = self.lin(x)
        return x

# Load the dataloaders
def load_dataloaders():
    with open('train_dl.pth', 'rb') as f:
        train_dl = torch.load(f)
    with open('val_dl.pth', 'rb') as f:
        val_dl = torch.load(f)
    return train_dl, val_dl

train_dl, val_dl = load_dataloaders()

# Create the model and put it on the GPU if available
myModel = AudioClassifier()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
myModel = myModel.to(device)
print(next(myModel.parameters()).device)

def training(model, train_dl):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                    max_lr=0.001,
                                                    steps_per_epoch=int(len(train_dl)),
                                                    epochs=1,
                                                    anneal_strategy='linear')
    epoch_losses = []
    epoch_accuracies = []
    epoch = 0

    try:
        while True:
            running_loss = 0.0
            correct_prediction = 0
            total_prediction = 0
            for i, data in enumerate(train_dl):
                inputs, labels = data[0].to(device), data[1].to(device)
                inputs_m, inputs_s = inputs.mean(), inputs.std()
                inputs = (inputs - inputs_m) / inputs_s
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                scheduler.step()
                running_loss += loss.item()
                _, prediction = torch.max(outputs, 1)
                correct_prediction += (prediction == labels).sum().item()
                total_prediction += prediction.shape[0]
                if i % 10 == 0:
                    print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 10))

            num_batches = len(train_dl)
            avg_loss = running_loss / num_batches
            acc = correct_prediction / total_prediction
            print(f'Epoch: {epoch}, Loss: {avg_loss:.2f}, Accuracy: {acc:.2f}')

            epoch_losses.append(avg_loss)
            epoch_accuracies.append(acc)

            epoch += 1
    except KeyboardInterrupt:
        print("Training interrupted. Saving model...")

    print('Done Training')
    return epoch_losses, epoch_accuracies

epoch_losses, epoch_accuracies = training(myModel, train_dl)

# Save the trained model
torch.save(myModel.state_dict(), 'bird_classifier_model.pth')

# Plotting the training loss and accuracy
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(range(len(epoch_losses)), epoch_losses, label='Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(len(epoch_accuracies)), epoch_accuracies, label='Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training Accuracy')
plt.legend()

plt.show()

def inference(model, val_dl):
    correct_prediction = 0
    total_prediction = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for data in val_dl:
            inputs, labels = data[0].to(device), data[1].to(device)
            inputs_m, inputs_s = inputs.mean(), inputs.std()
            inputs = (inputs - inputs_m) / inputs_s
            outputs = model(inputs)
            _, prediction = torch.max(outputs, 1)
            correct_prediction += (prediction == labels).sum().item()
            total_prediction += prediction.shape[0]

            all_preds.extend(prediction.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = correct_prediction / total_prediction
    print(f'Accuracy: {acc:.2f}, Total items: {total_prediction}')

    # Compute and plot confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

    return all_preds, all_labels

# Get species names from the prepared metadata file
prepared_metadata_file = Path.cwd()/'BirdSongsEurope'/'prepared_metadata.csv'
df = pd.read_csv(prepared_metadata_file)
species_names = df['Species'].unique()

# Run inference and get predictions
all_preds, all_labels = inference(myModel, val_dl)

# Calculate species-wise accuracy
species_accuracy = {}
for species_name in species_names:
    species_id = df[df['Species'] == species_name].index[0]  # Get species ID based on its first occurrence
    species_mask = (all_labels == species_id)
    species_correct = (all_preds[species_mask] == species_id).sum()
    species_total = species_mask.sum()
    species_accuracy[species_name] = species_correct / species_total if species_total > 0 else 0

# Plot species-wise accuracy
plt.figure(figsize=(15, 8))
plt.bar(range(len(species_accuracy)), list(species_accuracy.values()), align='center')
plt.xticks(range(len(species_accuracy)), list(species_accuracy.keys()), rotation=90)
plt.xlabel('Species')
plt.ylabel('Accuracy')
plt.title('Species-wise Accuracy')
plt.show()
