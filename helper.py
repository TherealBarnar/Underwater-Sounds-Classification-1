import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import Counter
import librosa
import os
import numpy as np

df = pd.read_csv("audio_features_dataset_no_duplicates.csv")


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
    plt.figure(figsize=(20,10))
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
    plt.figure(figsize=(20,10))
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
    bars_target = plt.bar(frequency_counts_target.index.astype(str), frequency_counts_target.values, color='skyblue', label='Target')

    # Grafico per dfNoTarget
    bars_no_target = plt.bar(frequency_counts_no_target.index.astype(str), frequency_counts_no_target.values, color='lightcoral', label='Non-Target')

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
    plt.axvline(overall_median, color='red', linestyle='--', linewidth=2, label=f'Median Duration: {overall_median:.2f}')

    # Aggiungi titolo e label agli assi
    plt.title('Distribution of Durations for Target and No Target Classes')
    plt.xlabel('Duration (seconds)')
    plt.ylabel('Count')
    plt.legend()
    plt.show()




