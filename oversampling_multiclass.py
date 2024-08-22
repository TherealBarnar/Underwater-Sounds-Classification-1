import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import numpy as np

def split_dataset(csv_file, train_size=0.8, val_size=0.1, test_size=0.1):
    """
    Carica un dataset CSV, lo divide in set di addestramento, validazione e test,
    e restituisce le caratteristiche e i target separati.

    Args:
        csv_file (str): Percorso al file CSV del dataset.
        train_size (float): Percentuale del dataset da utilizzare per il training.
        val_size (float): Percentuale del dataset da utilizzare per la validazione.
        test_size (float): Percentuale del dataset da utilizzare per il test.

    Returns:
        tuple: (X_train_imputed, X_val_imputed, X_test_imputed, y_train_encoded, y_val_encoded, y_test_encoded)
            - X_train_imputed (ndarray): Caratteristiche del set di addestramento dopo imputazione.
            - X_val_imputed (ndarray): Caratteristiche del set di validazione dopo imputazione.
            - X_test_imputed (ndarray): Caratteristiche del set di test dopo imputazione.
            - y_train_encoded (ndarray): Target del set di addestramento codificato.
            - y_val_encoded (ndarray): Target del set di validazione codificato.
            - y_test_encoded (ndarray): Target del set di test codificato.
    """
    # Carica il CSV
    df = pd.read_csv(csv_file)

    # Estrai il nome base del file audio (es. 'audioA')
    df['BaseFileName'] = df['File Name'].str.split('_seg').str[0]

    # Raggruppa per 'BaseFileName'
    file_names = df['BaseFileName'].unique()

    # Suddividi i nomi dei file in train, validation e test
    train_file_names, test_file_names = train_test_split(file_names, test_size=0.2, random_state=42)
    val_file_names, test_file_names = train_test_split(test_file_names, test_size=0.5, random_state=42)

    # Filtra i dati per ciascun set
    train_df = df[df['BaseFileName'].isin(train_file_names)]
    val_df = df[df['BaseFileName'].isin(val_file_names)]
    test_df = df[df['BaseFileName'].isin(test_file_names)]

    # Rimuovi le colonne non necessarie
    train_df = train_df.drop(columns=['BaseFileName', 'File Name'])
    val_df = val_df.drop(columns=['BaseFileName', 'File Name'])
    test_df = test_df.drop(columns=['BaseFileName', 'File Name'])

    # Filtro per includere solo i campioni con la classe 'Target'
    train_df_target = train_df[train_df['Class'] == 'Target']
    val_df_target = val_df[val_df['Class'] == 'Target']
    test_df_target = test_df[test_df['Class'] == 'Target']

    # Separa le caratteristiche e il target
    X_train = train_df_target.drop(columns=['Class', 'Subclass'])
    y_train = train_df_target['Subclass']
    X_val = val_df_target.drop(columns=['Class', 'Subclass'])
    y_val = val_df_target['Subclass']
    X_test = test_df_target.drop(columns=['Class', 'Subclass'])
    y_test = test_df_target['Subclass']

    # Codifica le etichette "Subclass" in numeri interi
    subclass_encoder = LabelEncoder()
    y_train_encoded = subclass_encoder.fit_transform(y_train)
    y_val_encoded = subclass_encoder.transform(y_val)
    y_test_encoded = subclass_encoder.transform(y_test)

    # Imputazione dei valori mancanti
    imputer = SimpleImputer(strategy='mean')
    X_train_imputed = imputer.fit_transform(X_train)
    X_val_imputed = imputer.transform(X_val)
    X_test_imputed = imputer.transform(X_test)

    return X_train_imputed, X_val_imputed, X_test_imputed, y_train_encoded, y_val_encoded, y_test_encoded, subclass_encoder

def apply_smote(X_train, y_train, k_neighbors=5):
    """
    Applica SMOTE al set di addestramento per bilanciare le classi.

    Args:
        X_train (ndarray): Caratteristiche del set di addestramento.
        y_train (ndarray): Target del set di addestramento.
        k_neighbors (int): Numero di vicini da considerare per SMOTE.

    Returns:
        tuple: (X_train_resampled, y_train_resampled)
            - X_train_resampled (ndarray): Caratteristiche del set di addestramento dopo SMOTE.
            - y_train_resampled (ndarray): Target del set di addestramento dopo SMOTE.
    """
    # Crea l'istanza di SMOTE
    smote = SMOTE(sampling_strategy='auto', k_neighbors=k_neighbors, random_state=42)

    # Trova le classi con campioni sufficienti
    class_counts = pd.Series(y_train).value_counts()
    min_samples_per_class = k_neighbors  # Numero minimo di campioni per usare SMOTE

    # Escludi classi con meno di min_samples_per_class campioni
    classes_to_keep = class_counts[class_counts >= min_samples_per_class].index
    mask = np.isin(y_train, classes_to_keep)
    X_train_filtered = X_train[mask]
    y_train_filtered = y_train[mask]

    # Applica SMOTE solo al set di addestramento filtrato
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_filtered, y_train_filtered)

    return X_train_resampled, y_train_resampled

