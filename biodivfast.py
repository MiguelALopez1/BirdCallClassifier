import os
import librosa
import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from collections import Counter
from tqdm import tqdm
from joblib import Parallel, delayed, dump, load

# Load and preprocess audio files
def load_audio_files(audio_dir):
    audio_files = []
    for file in os.listdir(audio_dir):
        if file.endswith('.wav'):
            audio_files.append(os.path.join(audio_dir, file))
    return audio_files

def preprocess_audio(file_path):
    y, sr = librosa.load(file_path, sr=None, mono=True)
    return y, sr

audio_dir = 'Data'
audio_files = load_audio_files(audio_dir)

# Break down preprocessing into batches and save intermediate results
batch_size = 50
preprocessed_audio = []

for i in range(0, len(audio_files), batch_size):
    batch_files = audio_files[i:i+batch_size]
    batch_preprocessed = Parallel(n_jobs=-1)(
        delayed(preprocess_audio)(file) for file in tqdm(batch_files, desc=f"Preprocessing Audio Batch {i//batch_size + 1}", unit="file")
    )
    batch_path = f"preprocessed_audio_batch_{i//batch_size + 1}.pkl"
    dump(batch_preprocessed, batch_path)
    preprocessed_audio.extend(batch_preprocessed)

# Extract features in batches and save intermediate results
def extract_features(y, sr):
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    delta_mfccs = librosa.feature.delta(mfccs)
    delta2_mfccs = librosa.feature.delta(mfccs, order=2)
    features = np.vstack([mfccs, delta_mfccs, delta2_mfccs])
    return features.T

all_features = []

for i in range(0, len(preprocessed_audio), batch_size):
    batch_preprocessed = preprocessed_audio[i:i+batch_size]
    batch_features = Parallel(n_jobs=-1)(
        delayed(extract_features)(y, sr) for y, sr in tqdm(batch_preprocessed, desc=f"Extracting Features Batch {i//batch_size + 1}", unit="file")
    )
    batch_path = f"features_batch_{i//batch_size + 1}.pkl"
    dump(batch_features, batch_path)
    all_features.extend(batch_features)

# Concatenate all features
all_features = np.concatenate(all_features, axis=0)

# Determine number of clusters using Elbow Method
def plot_elbow_method(features):
    distortions = []
    K = range(1, 20)
    for k in tqdm(K, desc="Elbow Method", unit="cluster"):
        kmeans = KMeans(n_clusters=k, n_init=10)
        kmeans.fit(features)
        distortions.append(kmeans.inertia_)
    plt.figure(figsize=(16, 8))
    plt.plot(K, distortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Distortion')
    plt.title('The Elbow Method showing the optimal k')
    plt.show()

plot_elbow_method(all_features)

# Alternatively, use AIC/BIC with GMM
def find_optimal_gmm(features):
    lowest_bic = np.infty
    bic = []
    n_components_range = range(1, 20)
    for n_components in tqdm(n_components_range, desc="GMM Clustering", unit="component"):
        gmm = GaussianMixture(n_components=n_components, n_init=1)
        gmm.fit(features)
        bic.append(gmm.bic(features))
        if bic[-1] < lowest_bic:
            lowest_bic = bic[-1]
            best_gmm = gmm
    plt.figure(figsize=(16, 8))
    plt.plot(n_components_range, bic, 'bx-')
    plt.xlabel('Number of components')
    plt.ylabel('BIC')
    plt.title('BIC for GMM clustering')
    plt.show()
    return best_gmm

best_gmm = find_optimal_gmm(all_features)

# Perform clustering
optimal_k = 3  # or 4 based on the elbow point
kmeans = KMeans(n_clusters=optimal_k, n_init=10)
kmeans.fit(all_features)
labels = kmeans.labels_

# Compute Simpson Biodiversity Index
def compute_simpson_index(labels):
    counts = Counter(labels)
    N = sum(counts.values())
    simpson_index = 1 - sum((count * (count - 1)) / (N * (N - 1)) for count in counts.values())
    return simpson_index

simpson_index = compute_simpson_index(labels)
print(f"Estimated Simpson Biodiversity Index: {simpson_index}")
