import os
import librosa
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from scipy.stats import entropy

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

# Funzione per calcolare l'energia media dei MFCC
def calculate_mfcc_mean_energy(y, sr):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=1)
    return np.mean(mfcc)

# Funzione per calcolare i passaggi di soglia
def calculate_threshold_crossings(y, threshold=0.01):
    return np.sum(np.abs(np.diff(y > threshold)))

# Funzione per calcolare il rapporto di silenzio
def calculate_silence_ratio(y, threshold=0.01):
    return np.sum(y < threshold) / len(y)

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
        mfcc_mean_energy = calculate_mfcc_mean_energy(y, sr)
        threshold_crossings = calculate_threshold_crossings(y)
        silence_ratio = calculate_silence_ratio(y)

        return [
            spectral_centroid_mean,
            spectral_bandwidth_rms,
            std_dev,
            skewness,
            kurt,
            shannon_entropy,
            mfcc_mean_energy,
            threshold_crossings,
            silence_ratio
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
    for root, _, files in os.walk(audio_directory):
        for filename in files:
            file_path = os.path.join(root, filename)
            if filename.endswith('.wav'):
                print(f"Processing file: {filename}")  # Stampa il nome del file in elaborazione
                features = extract_features(file_path)
                if features is not None:
                    features_list.append(features)
                    file_names.append(filename)
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
        'MFCC Mean Energy',
        'Threshold Crossings',
        'Silence Ratio'
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
