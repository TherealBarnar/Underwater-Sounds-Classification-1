import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import Counter
import librosa
import os
import numpy as np


df = pd.read_csv("audio_features_dataset_no_duplicates.csv")


file_audio = 'C://Users//mario//OneDrive//Desktop//Dataset - senza_duplicati//Non-Target//Right whale//RightWhale.wav'
cargo_ship = "C://Users//mario//OneDrive//Desktop//Cargo-Ship-at-20-knts.wav"

def estrai_classe_target(file_path):
    # Filtra le righe dove la colonna 'classe' è 'Target'
    df_target = df[df['Classe'] == 'Target']

    return df_target


def estrai_classe_noTarget(file_path):
    df_notarget = df[df['Classe'] == 'Non-Target']
    return df_notarget


dfTarget = estrai_classe_target(df)
dfNoTarget = estrai_classe_noTarget(df)


def sample_counter(dfTarget):
    category_counts = dfTarget['Sottoclasse'].value_counts()
    plt.figure(figsize=(20, 10))
    ax = sns.barplot(x=category_counts.index, y=category_counts.values)
    plt.xlabel('Categoria')
    plt.ylabel('Numero di elementi')
    plt.title('Sample for each category')
    plt.xticks(rotation=45)

    for i, count in enumerate(category_counts):
        ax.text(i, count + 0.5, str(count), ha='center', va='bottom', fontsize=10)

    print("samples for Target Class:\n" + str(category_counts))
    plt.tight_layout()
    plt.show()


def sample_counterNot(dfNoTarget):
    category_counts = dfNoTarget['Sottoclasse'].value_counts()
    plt.figure(figsize=(20, 10))
    ax = sns.barplot(x=category_counts.index, y=category_counts.values)
    plt.xlabel('Categoria')
    plt.ylabel('Numero di elementi')
    plt.title('Sample for each category')
    plt.xticks(rotation=45)

    for i, count in enumerate(category_counts):
        ax.text(i, count + 0.5, str(count), ha='center', va='bottom', fontsize=10)

    print("samples for Non-Target Class:\n" + str(category_counts))
    plt.tight_layout()
    plt.show()


def plot_frequency_distribution(dfTarget):
    # Rimuovi eventuali valori NaN risultanti dalla conversione
    dfTarget = dfTarget.dropna(subset=['Frequenza'])

    # Conta le occorrenze di ciascuna frequenza
    frequency_counts = dfTarget['Frequenza'].value_counts().sort_index()

    sr_max = dfTarget['Frequenza'].max()
    sr_min = dfTarget['Frequenza'].min()

    print(f"Sample Rate Massimo: {sr_max}")
    print(f"Sample Rate Minimo: {sr_min}")

    # Crea il grafico a barre
    plt.figure(figsize=(12, 6))
    bars = plt.bar(frequency_counts.index.astype(str), frequency_counts.values, color='skyblue')

    # Aggiungi i numeri sopra le barre
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height, f'{int(height)}', ha='center', va='bottom')

    plt.title('Distribuzione delle Frequenze per la Classe Target')
    plt.xlabel('Frequenza')
    plt.ylabel('Occorrenze')
    plt.xticks(rotation=45)
    plt.tight_layout()  # Aggiunge spaziatura per evitare sovrapposizioni
    plt.show()


def plot_frequency_distribution_not(dfNoTarget):
    # Rimuovi eventuali valori NaN risultanti dalla conversione
    dfNoTarget = dfNoTarget.dropna(subset=['Frequenza'])

    # Conta le occorrenze di ciascuna frequenza
    frequency_counts = dfNoTarget['Frequenza'].value_counts().sort_index()

    sr_max = dfNoTarget['Frequenza'].max()
    sr_min = dfNoTarget['Frequenza'].min()

    print(f"Sample Rate Massimo: {sr_max}")
    print(f"Sample Rate Minimo: {sr_min}")

    # Crea il grafico a barre
    plt.figure(figsize=(12, 6))
    bars = plt.bar(frequency_counts.index.astype(str), frequency_counts.values, color='skyblue')

    # Aggiungi i numeri sopra le barre
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height, f'{int(height)}', ha='center', va='bottom')

    plt.title('Distribuzione delle Frequenze per la Classe Non-Target')
    plt.xlabel('Frequenza')
    plt.ylabel('Occorrenze')
    plt.xticks(rotation=45)
    plt.tight_layout()  # Aggiunge spaziatura per evitare sovrapposizioni
    plt.show()


def plot_combined_frequency_distribution(dfTarget, dfNoTarget):
    # Rimuovi eventuali valori NaN risultanti dalla conversione per il dataframe dfTarget
    dfTarget = dfTarget.dropna(subset=['Frequenza'])

    # Conta le occorrenze di ciascuna frequenza per dfTarget
    frequency_counts_target = dfTarget['Frequenza'].value_counts().sort_index()

    # Rimuovi eventuali valori NaN risultanti dalla conversione per il dataframe dfNoTarget
    dfNoTarget = dfNoTarget.dropna(subset=['Frequenza'])

    # Conta le occorrenze di ciascuna frequenza per dfNoTarget
    frequency_counts_no_target = dfNoTarget['Frequenza'].value_counts().sort_index()

    # Crea il grafico a barre
    plt.figure(figsize=(12, 6))

    # Grafico per dfTarget
    bars_target = plt.bar(frequency_counts_target.index.astype(str), frequency_counts_target.values, color='skyblue',
                          label='Target')

    # Grafico per dfNoTarget
    bars_no_target = plt.bar(frequency_counts_no_target.index.astype(str), frequency_counts_no_target.values,
                             color='lightcoral', label='Non-Target')

    # Aggiungi i numeri sopra le barre per dfTarget
    for bar in bars_target:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height, f'{int(height)}', ha='center', va='bottom')

    # Aggiungi i numeri sopra le barre per dfNoTarget
    for bar in bars_no_target:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height, f'{int(height)}', ha='center', va='bottom')

    plt.title('Distribuzione delle Frequenze per Target e Non-Target')
    plt.xlabel('Frequenza')
    plt.ylabel('Occorrenze')
    plt.xticks(rotation=45)
    plt.legend()  # Aggiungi la legenda con i colori corrispondenti
    plt.tight_layout()  # Aggiunge spaziatura per evitare sovrapposizioni
    plt.show()


def plot_duration_target(dfTarget):
    # Copia del DataFrame per evitare modifiche dirette
    dfTarget = dfTarget.copy()

    num_bins = 10  # Numero di intervalli desiderato
    bin_width = 250  # Larghezza dell'intervallo

    # Calcola gli estremi degli intervalli
    min_duration = dfTarget['Durata'].min()
    max_duration = dfTarget['Durata'].max()
    bin_edges = [0, 250, 500, 750, 1000, 1250, 1500, 2000]
    labels = ['0-250', '250-500', '500-750', '750-1000', '1000-1250', '1250-1500', '1750-2000']

    print(f"Max duration: {max_duration}")
    print(f"Min duration: {min_duration}")

    # Raggruppa le durate in base agli intervalli definiti
    dfTarget['Durata'] = pd.cut(dfTarget['Durata'], bins=bin_edges, right=False)

    # Conta le occorrenze di ciascun intervallo di durata
    duration_counts = dfTarget['Durata'].value_counts().sort_index()

    # Crea il grafico a barre
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(duration_counts)), duration_counts.values, color='skyblue')

    # Aggiungi le etichette agli intervalli sull'asse x
    plt.xticks(range(len(duration_counts)), duration_counts.index, rotation=45)

    # Aggiungi i numeri sopra le barre
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height, f'{int(height)}', ha='center', va='bottom')

    plt.title('Distribuzione delle Durate per la Classe Target')
    plt.xlabel('Durata')
    plt.ylabel('Occorrenze')
    plt.tight_layout()  # Aggiunge spaziatura per evitare sovrapposizioni
    plt.show()


def plot_combined_duration_distribution(dfTarget, dfNoTarget):
    # Calcola le durate per le classi dfTarget e dfNoTarget
    durations_target = dfTarget['Durata'].values
    durations_notarget = dfNoTarget['Durata'].values

    # Calcola la mediana complessiva
    all_durations = pd.concat([dfTarget['Durata'], dfNoTarget['Durata']])
    overall_median = all_durations.median()
    plt.figure(figsize=(10, 6))

    # Plot delle durate per dfTarget
    plt.hist(durations_target, bins=30, alpha=0.7, label='Target', color='skyblue')

    # Plot delle durate per dfNoTarget
    plt.hist(durations_notarget, bins=30, alpha=0.7, label='No Target', color='lightcoral')

    # Linea tratteggiata per la mediana complessiva
    plt.axvline(overall_median, color='red', linestyle='--', linewidth=2,
                label=f'Median Duration: {overall_median:.2f}')

    # Aggiungi titolo e label agli assi
    plt.title('Distribution of Durations for Target and No Target Classes')
    plt.xlabel('Duration (seconds)')
    plt.ylabel('Count')
    plt.legend()
    plt.show()


def prep_plot_channel_distribution(dataframe):
    # Contare le occorrenze dei diversi numeri di canali
    channel_counts = dataframe['Numero di canali'].value_counts().sort_index()

    # Creare il grafico
    plt.figure(figsize=(10, 6))
    bars = plt.bar(channel_counts.index, channel_counts.values, color='lightgreen')
    plt.bar(channel_counts.index, channel_counts.values, color='lightgreen')
    plt.xlabel('Numero di Canali')
    plt.ylabel('Occorrenze')
    plt.title('Distribuzione del Numero di Canali')
    plt.xticks(channel_counts.index)  # Assicurarsi che tutti i valori degli x siano mostrati
    plt.grid(axis='y')

    for bar, occorrenze in zip(bars, channel_counts.values):
        plt.annotate(occorrenze,
                     xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                     xytext=(0, 3),
                     textcoords="offset points",
                     ha='center', va='bottom', fontsize=9)
    plt.show()


def prep_plot_bit_depth_distribution(dataframe):
    # Contare le occorrenze dei diversi bit depth
    bit_depth_counts = dataframe['Bit Depth'].value_counts().sort_index()

    # Creare il grafico
    plt.figure(figsize=(10, 6))
    bars = plt.bar(bit_depth_counts.index, bit_depth_counts.values, color='lightgreen')
    plt.xlabel('Bit Depth')
    plt.ylabel('Occorrenze')
    plt.title('Distribuzione dei valori di Bit Depth')
    plt.xticks(bit_depth_counts.index)  # Assicurarsi che tutti i valori degli x siano mostrati
    plt.grid(axis='y')

    # Aggiungere etichette con il numero di occorrenze sopra le barre
    for bar, occorrenze in zip(bars, bit_depth_counts.values):
        plt.annotate(occorrenze,
                     xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                     xytext=(0, 3),  # Posizione dell'etichetta (3 punti sopra la barra)
                     textcoords="offset points",
                     ha='center', va='bottom', fontsize=9)

    # Mostrare il grafico
    plt.tight_layout()
    plt.show()


def create_spectrogram(file_path, sr):

    y, sr = librosa.load(file_path, sr=sr)

    # Calcola lo spettrogramma della potenza
    S = librosa.feature.melspectrogram(y=y, sr=sr)

    # Converte in decibel
    S_db = librosa.power_to_db(S, ref=np.max)

    # Mostra lo spettrogramma della potenza con sfondo nero
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='mel', cmap='magma')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'Spectrogram (Sample Rate: {sr} Hz)')
    plt.tight_layout()
    plt.show()


def create_spectrogram_low_sr(file_path, sr):
    # Carica il file audio con la frequenza di campionamento specificata
    y, _ = librosa.load(file_path, sr=sr)

    # Calcola lo spettrogramma della potenza con parametri aggiustati
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64, fmax=8000)

    # Converte in decibel
    S_db = librosa.power_to_db(S, ref=np.max)

    # Mostra lo spettrogramma della potenza con sfondo nero
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='mel', cmap='magma')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'Spectrogram (Sample Rate: {sr} Hz)')
    plt.tight_layout()
    plt.show()



def plot_audio_waveform(file_path):

    try:
        # Carica il file audio utilizzando librosa
        y, sr = librosa.load(file_path, sr=None, mono=True)

        # Calcola il tempo in secondi per ciascun campione
        time = np.linspace(0, len(y) / sr, len(y))

        # Crea il grafico dell'ampiezza del segnale audio
        plt.figure(figsize=(12, 6))
        plt.plot(time, y, label='Ampiezza')
        plt.title(f'Forma d\'onda del file audio: {file_path}')
        plt.xlabel('Tempo (s)')
        plt.ylabel('Ampiezza')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Errore durante l'elaborazione del file '{file_path}': {e}")



if __name__ == "__main__":
    plot_audio_waveform(cargo_ship)
