import os
import librosa
import pandas as pd
import matplotlib.pyplot as plt

dataset_root = 'C://underwater-classification//dataset//Target'


def extract_audio_features(dataset_root):
    # Lista per memorizzare le informazioni estratte
    audio_features = []

    # Liste per memorizzare i valori delle caratteristiche
    durations = []
    frequencies = []
    amplitudes = []
    num_channels_list = []
    phases = []

    # Attraversa tutte le sottocartelle nel percorso radice
    for root, dirs, files in os.walk(dataset_root):
        for filename in files:
            if filename.endswith(('.wav', '.mp3')):  # Considera solo file audio WAV o MP3
                file_path = os.path.join(root, filename)

                try:
                    # Carica il file audio utilizzando librosa
                    y, sr = librosa.load(file_path, sr=None)

                    # Estrai le informazioni richieste
                    amplitude = max(abs(y))  # Ampiezza massima assoluta del segnale
                    duration = librosa.get_duration(y=y, sr=sr)  # Durata del segnale in secondi
                    num_channels = 1 if y.ndim == 1 else y.shape[1]  # Numero di canali (mono o stereo)
                    frequency = sr  # Frequenza di campionamento
                    phase = y[0]  # Fase del segnale (primo campione)

                    # Aggiungi i valori alle liste corrispondenti
                    amplitudes.append(amplitude)
                    durations.append(duration)
                    frequencies.append(frequency)
                    num_channels_list.append(num_channels)
                    phases.append(phase)

                    # Ottieni il nome del file senza l'estensione
                    file_name = os.path.splitext(filename)[0]

                    # Aggiungi le informazioni estratte alla lista
                    audio_features.append({
                        'Nome file': file_name,
                        'Path del file': file_path,
                        'Ampiezza del segnale': amplitude,
                        'Durata': duration,
                        'Frequenza': frequency,
                        'Numero di canali': num_channels,
                        'Fase': phase,
                        'Forma d\'onda': y.shape
                    })

                except Exception as e:
                    print(f"Errore durante l'elaborazione del file '{filename}': {e}")

    return audio_features, amplitudes, durations, frequencies, num_channels_list, phases

def save_to_csv(data, output_file):
    # Converte i dati in un DataFrame pandas
    df = pd.DataFrame(data)

    # Salva il DataFrame in un file CSV
    df.to_csv(output_file, index=False)

def plot_distribution(values, title):
    # Crea un grafico di distribuzione utilizzando matplotlib
    plt.figure(figsize=(8, 6))
    plt.hist(values, bins=20, color='skyblue', edgecolor='black')
    plt.title(title)
    plt.xlabel('Valore')
    plt.ylabel('Frequenza')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # Specifica il percorso radice del dataset audio


    # Estrai le informazioni audio dal dataset
    extracted_features, amplitudes, durations, frequencies, num_channels_list, phases = extract_audio_features(dataset_root)

    # Specifica il percorso per salvare il file CSV di output
    output_csv_file = 'audio_features_dataset.csv'

    # Salva le informazioni estratte in un file CSV
    save_to_csv(extracted_features, output_csv_file)

    # Genera i grafici di distribuzione per ciascuna caratteristica
    plot_distribution(amplitudes, 'Distribuzione dei Valori di Ampiezza del Segnale')
    plot_distribution(durations, 'Distribuzione dei Valori di Durata')
    plot_distribution(frequencies, 'Distribuzione dei Valori di Frequenza')
    plot_distribution(num_channels_list, 'Distribuzione dei Valori di Numero di Canali')
    plot_distribution(phases, 'Distribuzione dei Valori di Fase')

    print(f"Il file CSV '{output_csv_file}' e' stato creato con successo.")
