import librosa
import librosa.display
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import Counter

def plot_audio_durations_histogram(durations, bins=100):
    plt.figure(figsize=(12, 8))
    sns.histplot(durations, bins=bins, kde=True)
    plt.title('Distribuzione delle durate degli audio')
    plt.xlabel('Durata (secondi)')
    plt.ylabel('Frequenza')
    plt.grid(True)
    plt.show()

def plot_audio_max_frequencies_histogram(max_internal_frequencies, bins=100):
    plt.figure(figsize=(12, 8))
    sns.histplot(max_internal_frequencies, bins=bins, kde=True)
    plt.title('Distribuzione delle frequenze massime')
    plt.xlabel('Frequenza massima')
    plt.ylabel('Valore')
    plt.grid(True)
    plt.show()

def plot_distribution_boxplot(values, title):
    plt.figure(figsize=(10, 6))
    plt.boxplot(values, vert=False)  # Imposta vert=False per visualizzare il box plot orizzontalmente
    plt.title(title)
    plt.xlabel('Valore')
    plt.grid(True)
    plt.show()

def plot_distribution(values, title, x_label):
    # Conta le occorrenze di ciascun valore
    counter = Counter(values)

    # Ordina i valori unici
    unique_values = sorted(counter.keys())
    counts = [counter[value] for value in unique_values]

    # Crea un grafico di distribuzione utilizzando matplotlib
    plt.figure(figsize=(12, 8))
    bar_width = 0.8  # Larghezza delle barre
    indices = range(len(unique_values))

    bars = plt.bar(indices, counts, color='skyblue', edgecolor='black', width=bar_width)

    # Aggiunge le etichette alle barre
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height, str(count), ha='center', va='bottom', fontsize=10)

    plt.title(title, fontsize=16)
    plt.xlabel(x_label, fontsize=14)
    plt.ylabel('Occorrenze', fontsize=14)
    plt.xticks(indices, unique_values, rotation=45, ha='right')
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Carica i dati dal file CSV
    input_csv_file = 'audio_features_dataset.csv'
    df = pd.read_csv(input_csv_file)

    # Estrai le liste delle caratteristiche
    amplitudes = df['Ampiezza del segnale'].tolist()
    durations = df['Durata'].tolist()
    frequencies = df['Frequenza'].tolist()
    num_channels_list = df['Numero di canali'].tolist()
    phases = df['Fase'].tolist()
    max_internal_frequencies = df['Frequenza massima interna'].tolist()
    bit_depths = df['Bit Depth'].tolist()

    # Genera i grafici di distribuzione per ciascuna caratteristica
    plot_distribution_boxplot(amplitudes, 'Distribuzione dei Valori di Ampiezza del Segnale')
    plot_audio_durations_histogram(durations)
    plot_distribution(frequencies, 'Distribuzione dei Valori di Frequenza', 'Frequenza (Hz)')
    plot_distribution(num_channels_list, 'Distribuzione dei Valori di Numero di Canali', 'Numero di Canali')
    plot_distribution_boxplot(phases, 'Distribuzione dei Valori di Fase')
    plot_audio_max_frequencies_histogram(max_internal_frequencies)
    plot_distribution(bit_depths, 'Distribuzione dei Valori di Bit Depth', 'Bit Depth')

