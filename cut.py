import os
import librosa
import soundfile as sf
import pandas as pd

def extract_audio_features(root_folder):
    audio_features = []
    segment_features = []  # Lista per memorizzare le features dei segmenti
    num_segments_per_file = 0  # Contatore per il numero di segmenti

    for root, dirs, files in os.walk(root_folder):
        for filename in files:
            if filename.endswith(('.wav', '.mp3', '.flac')):
                file_path = os.path.join(root, filename)

                try:
                    file_path = str(file_path)

                    # Carica il file audio utilizzando librosa in mono
                    y, sr = librosa.load(file_path, sr=None, mono=True)

                    # Debugging print statement before extraction
                    print(f"Processing file: {file_path}")
                    print(f"Loaded audio shape: {y.shape}, dtype: {y.dtype}")

                    with sf.SoundFile(file_path) as f:
                        num_channels = f.channels
                        bit_depth = ' '.join(sf.info(file_path).subtype_info.split())

                    amplitude = max(abs(y))
                    duration = librosa.get_duration(y=y, sr=sr)
                    frequency = sr
                    phase = y[0]

                    pitches, magnitudes = librosa.core.piptrack(y=y, sr=sr, n_fft=512)
                    max_idx = magnitudes.argmax()
                    max_time_idx = max_idx // magnitudes.shape[0]
                    max_freq_idx = max_idx % magnitudes.shape[0]
                    max_internal_frequency = pitches[max_freq_idx, max_time_idx]

                    amplitudes = [amplitude]  # Lista per le ampiezze dei segmenti
                    durations = [duration]  # Lista per le durate dei segmenti
                    frequencies = [frequency]  # Lista per le frequenze dei segmenti

                    # Calcola il numero di segmenti
                    num_segments = int(duration // 4)
                    if duration % 4 != 0:
                        num_segments += 1

                    # Taglia e aggiungi ogni segmento
                    for i in range(num_segments):
                        start_time = i * 4
                        end_time = min((i + 1) * 4, duration)  # Gestione dell'ultimo segmento

                        # Calcola gli indici di campione per il taglio
                        start_sample = int(start_time * sr)
                        end_sample = int(end_time * sr)

                        # Esegui il taglio
                        y_cut = y[start_sample:end_sample]

                        # Aggiungi le feature del segmento alla lista
                        segment_features.append({
                            'Nome file': f"{os.path.splitext(filename)[0]}_seg{i + 1}",
                            'Classe': os.path.basename(os.path.dirname(os.path.dirname(file_path))),
                            'Sottoclasse': os.path.basename(os.path.dirname(file_path)),
                            'Ampiezza del segnale': max(abs(y_cut)),
                            'Durata': librosa.get_duration(y=y_cut, sr=sr),
                            'Frequenza': sr,
                            'Numero di canali': num_channels,
                            'Fase': y_cut[0],
                            'Frequenza massima interna': max(librosa.core.piptrack(y=y_cut, sr=sr, n_fft=512)[0]),
                            'Bit Depth': bit_depth,
                            'Forma d\'onda': y_cut.shape,
                            'Segmento Numero': i + 1,
                            'File di Provenienza': filename
                        })

                    # Aggiungi le feature del file originale alla lista
                    audio_features.append({
                        'Nome file': os.path.splitext(filename)[0],
                        'Classe': os.path.basename(os.path.dirname(os.path.dirname(file_path))),
                        'Sottoclasse': os.path.basename(os.path.dirname(file_path)),
                        'Ampiezza del segnale': amplitude,
                        'Durata': duration,
                        'Frequenza': frequency,
                        'Numero di canali': num_channels,
                        'Fase': phase,
                        'Frequenza massima interna': max_internal_frequency,
                        'Bit Depth': bit_depth,
                        'Forma d\'onda': y.shape
                    })

                except Exception as e:
                    print(f"Errore durante l'elaborazione del file '{file_path}': {e}")

    return audio_features + segment_features


def save_to_csv(data, output_file):
    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)

if __name__ == "__main__":
    dataset_root = "C://Users//mario//OneDrive//Desktop//Dataset_1//Target//Seal bomb//"

    # Estrai le features audio compresi i segmenti
    extracted_features = extract_audio_features(dataset_root)

    output_csv_file = 'audio_features_dataset_cut.csv'
    save_to_csv(extracted_features, output_csv_file)
    print(f"Il file CSV '{output_csv_file}' è stato creato con successo.")
