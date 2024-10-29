import os
import librosa
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from scipy.stats import entropy
from tqdm import tqdm  # Importa tqdm per la progress bar

# Funzione per calcolare il centroid spettrale
def calculate_spectral_centroid(y, sr):
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    return np.mean(spectral_centroid)

# Funzione per calcolare la larghezza di banda spettrale
def calculate_spectral_bandwidth(y, sr):
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    return np.mean(spectral_bandwidth)

# Funzione per calcolare la deviazione standard del segnale
def calculate_std_dev(y):
    return np.std(y)

# Funzione per calcolare la skewness
def calculate_skewness(y):
    return skew(y)

# Funzione per calcolare la kurtosis
def calculate_kurtosis(y):
    return kurtosis(y)

# Funzione per calcolare l'entropia di Shannon
def calculate_shannon_entropy(y):
    histogram, _ = np.histogram(y, bins=256, density=True)
    return entropy(histogram)

# Funzione per calcolare l'entropia di Renyi
def calculate_renyi_entropy(y, alpha=2):
    if alpha <= 0 or alpha == 1:
        raise ValueError("Alpha must be greater than 0 and not equal to 1")
    histogram, _ = np.histogram(y, bins=256, density=True)
    histogram = histogram[histogram > 0]  # Rimuove gli zeri
    return 1 / (1 - alpha) * np.log(np.sum(histogram ** alpha))

# Funzione per calcolare il tasso di attacco (rate of attack)
def calculate_rate_of_attack(y, sr):
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    return np.mean(librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr))

# Funzione per calcolare il tasso di decadimento (rate of decay)
def calculate_rate_of_decay(y):
    decay = np.diff(y)
    return np.mean(np.abs(decay[decay < 0]))

# Funzione per calcolare i passaggi di soglia
def calculate_threshold_crossings(y, threshold=0.01):
    return np.sum(np.abs(np.diff(y > threshold)))

# Funzione per calcolare il rapporto di silenzio
def calculate_silence_ratio(y, threshold=0.01):
    return np.sum(y < threshold) / len(y)

# Funzione per calcolare la media
def calculate_mean(y):
    return np.mean(y)

# Funzione per calcolare il massimo sulla media
def calculate_max_over_mean(y):
    return np.max(y) / np.mean(y)

# Funzione per calcolare il minimo sulla media
def calculate_min_over_mean(y):
    return np.min(y) / np.mean(y)

# Funzione per calcolare l'energia media
def calculate_energy_measurements(y):
    return np.sum(y ** 2) / len(y)

# Funzione per estrarre tutte le caratteristiche
def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=None)

        # Estrazione delle caratteristiche
        spectral_centroid_mean = calculate_spectral_centroid(y, sr)
        spectral_bandwidth_rms = calculate_spectral_bandwidth(y, sr)
        std_dev = calculate_std_dev(y)
        skewness = calculate_skewness(y)
        kurt = calculate_kurtosis(y)
        shannon_entropy = calculate_shannon_entropy(y)
        renyi_entropy = calculate_renyi_entropy(y)
        rate_of_attack = calculate_rate_of_attack(y, sr)
        rate_of_decay = calculate_rate_of_decay(y)
        threshold_crossings = calculate_threshold_crossings(y)
        silence_ratio = calculate_silence_ratio(y)
        mean = calculate_mean(y)
        max_over_mean = calculate_max_over_mean(y)
        min_over_mean = calculate_min_over_mean(y)
        energy_measurements = calculate_energy_measurements(y)

        return [
            spectral_centroid_mean,
            spectral_bandwidth_rms,
            std_dev,
            skewness,
            kurt,
            shannon_entropy,
            renyi_entropy,
            rate_of_attack,
            rate_of_decay,
            threshold_crossings,
            silence_ratio,
            mean,
            max_over_mean,
            min_over_mean,
            energy_measurements
        ]
    except Exception as e:
        print(f"Errore durante l'elaborazione di {file_path}: {str(e)}")
        return None

# Funzione per estrarre le caratteristiche da una directory e sottocartelle di file audio
def extract_features_from_directory(audio_directory):
    features_list = []
    file_names = []
    classes = []
    subclasses = []

    # Utilizzo di os.walk per esplorare ricorsivamente la directory e le sottocartelle
    files = []
    for root, _, filenames in os.walk(audio_directory):
        for filename in filenames:
            if filename.endswith('.wav'):
                files.append(os.path.join(root, filename))

    # Utilizzo di tqdm per la progress bar durante l'elaborazione di tutti i file
    for file_path in tqdm(files, desc="Processing audio files", unit="file"):
        features = extract_features(file_path)
        if features is not None:
            features_list.append(features)
            file_names.append(os.path.basename(file_path))
            # Ottieni classe e sottoclasse dalla struttura della directory
            rel_path = os.path.relpath(file_path, audio_directory)
            path_parts = rel_path.split(os.sep)
            class_name = path_parts[0] if len(path_parts) > 1 else 'Unknown'
            subclass_name = path_parts[1] if len(path_parts) > 2 else 'Unknown'
            classes.append(class_name)
            subclasses.append(subclass_name)
        else:
            print(f"Caratteristiche non estratte per {file_path}")

    columns = [
        'File Name',
        'Class',
        'Subclass',
        'Spectral Centroid Mean',
        'Spectral Bandwidth RMS',
        'Standard Deviation',
        'Skewness',
        'Kurtosis',
        'Shannon Entropy',
        'Renyi Entropy',
        'Rate of Attack',
        'Rate of Decay',
        'Threshold Crossings',
        'Silence Ratio',
        'Mean',
        'Max Over Mean',
        'Min Over Mean',
        'Energy Measurements'
    ]

    df = pd.DataFrame(features_list, columns=columns[3:])
    df.insert(0, 'File Name', file_names)
    df.insert(1, 'Class', classes)
    df.insert(2, 'Subclass', subclasses)

    return df

# Funzione principale (main)
def main():
    audio_directory = 'C:/Users/mario/OneDrive/Desktop/Dataset_oversampled'

    # Estrai le caratteristiche dalla directory
    df = extract_features_from_directory(audio_directory)

    if not df.empty:
        # Salva il DataFrame in un file CSV
        df.to_csv('audio_features.csv', index=False)
        print(f"Caratteristiche estratte da {len(df)} file audio e salvate in 'audio_features.csv'.")
    else:
        print("Nessuna caratteristica estratta. Verifica i file e riprova.")

# Chiamata alla funzione main se questo script è eseguito direttamente
if __name__ == "__main__":
    main()
