import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.svm import SVC
import lightgbm as lgb

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
        tuple: (X_train_imputed, X_val_imputed, X_test_imputed, y_train_encoded_filtered, y_val_encoded, y_test_encoded)
            - X_train_imputed (ndarray): Caratteristiche del set di addestramento dopo imputazione.
            - X_val_imputed (ndarray): Caratteristiche del set di validazione dopo imputazione.
            - X_test_imputed (ndarray): Caratteristiche del set di test dopo imputazione.
            - y_train_encoded_filtered (ndarray): Target del set di addestramento filtrato e codificato.
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

    # Separa le caratteristiche e il target
    X_train = train_df.drop(columns=['Class', 'Subclass'])
    y_train = train_df['Class']
    X_val = val_df.drop(columns=['Class', 'Subclass'])
    y_val = val_df['Class']
    X_test = test_df.drop(columns=['Class', 'Subclass'])
    y_test = test_df['Class']

    # Codifica le etichette "Class" (Target/Non-Target) in numeri binari
    class_encoder = LabelEncoder()
    y_train_encoded = class_encoder.fit_transform(y_train)
    y_val_encoded = class_encoder.transform(y_val)
    y_test_encoded = class_encoder.transform(y_test)

    # Verifica la distribuzione delle classi nel set di addestramento
    print("Distribuzione delle classi nel set di addestramento:")
    print(pd.Series(y_train_encoded).value_counts())

    # Impostazione della soglia per il numero minimo di campioni
    min_samples_threshold = 10

    # Trova le classi con abbastanza campioni
    classes_with_sufficient_samples = pd.Series(y_train_encoded).value_counts()
    classes_with_sufficient_samples = classes_with_sufficient_samples[
        classes_with_sufficient_samples >= min_samples_threshold].index

    # Filtra il set di addestramento per mantenere solo le classi con abbastanza campioni
    train_df_filtered = train_df[
        train_df['Class'].isin(class_encoder.inverse_transform(classes_with_sufficient_samples))]
    X_train_filtered = train_df_filtered.drop(columns=['Class', 'Subclass'])
    y_train_filtered = train_df_filtered['Class']

    # Codifica le etichette del set di addestramento filtrato
    y_train_encoded_filtered = class_encoder.transform(y_train_filtered)

    # Verifica la distribuzione delle classi dopo il filtro
    print("Distribuzione delle classi nel set di addestramento dopo il filtro:")
    print(pd.Series(y_train_encoded_filtered).value_counts())

    # Imputazione dei valori mancanti
    imputer = SimpleImputer(strategy='mean')
    X_train_imputed = imputer.fit_transform(X_train_filtered)
    X_val_imputed = imputer.transform(X_val)
    X_test_imputed = imputer.transform(X_test)

    return X_train_imputed, X_val_imputed, X_test_imputed, y_train_encoded_filtered, y_val_encoded, y_test_encoded


def apply_smote(X_train, y_train):
    """
    Applica SMOTE al set di addestramento per bilanciare le classi.

    Args:
        X_train (ndarray): Caratteristiche del set di addestramento.
        y_train (ndarray): Target del set di addestramento.

    Returns:
        tuple: (X_train_resampled, y_train_resampled)
            - X_train_resampled (ndarray): Caratteristiche del set di addestramento dopo SMOTE.
            - y_train_resampled (ndarray): Target del set di addestramento dopo SMOTE.
    """
    # Crea l'istanza di SMOTE
    smote = SMOTE(random_state=42)

    # Applica SMOTE solo al set di addestramento filtrato
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    # Verifica la distribuzione delle classi dopo SMOTE
    print(f"Distribuzione delle classi nel set di training dopo SMOTE:\n{pd.Series(y_train_resampled).value_counts()}")

    return X_train_resampled, y_train_resampled


def train_random_forest(X_train, y_train, X_val, y_val, n_steps=10):
    """
    Addestra un modello di Random Forest sul set di addestramento
    e valuta le sue prestazioni sul set di validazione.
    """
    # Inizializza il modello Random Forest
    rf_model = RandomForestClassifier(random_state=42)

    # Suddividi artificialmente il set di addestramento in step per simulare l'avanzamento
    data_size = len(X_train)
    step_size = data_size // n_steps

    for step in tqdm(range(n_steps), desc="Training Progress"):
        start = step * step_size
        end = (step + 1) * step_size if (step + 1) * step_size < data_size else data_size

        # Fit con un blocco incrementale dei dati
        rf_model.fit(X_train[:end], y_train[:end])

    # Previsioni sul set di validazione
    y_val_pred = rf_model.predict(X_val)

    # Controlla la distribuzione delle classi reali e predette
    print("Distribuzione delle classi reali:", np.bincount(y_val))
    print("Distribuzione delle classi predette:", np.bincount(y_val_pred))

    # Valuta il modello con il parametro zero_division per evitare l'avviso
    print("Report di classificazione del set di validazione:")
    print(classification_report(y_val, y_val_pred, zero_division=1))

    accuracy = accuracy_score(y_val, y_val_pred)
    print(f"Accuratezza sul set di validazione: {accuracy:.4f}")

    return rf_model


def train_svm(X_train, y_train, X_val, y_val):
    """
    Addestra un modello di Support Vector Machine (SVM) sul set di addestramento
    e valuta le sue prestazioni sul set di validazione.
    """
    # Inizializza il modello SVM
    svm_model = SVC(random_state=42)

    # Addestra il modello
    svm_model.fit(X_train, y_train)

    # Previsioni sul set di validazione
    y_val_pred = svm_model.predict(X_val)

    # Controlla la distribuzione delle classi reali e predette
    print("Distribuzione delle classi reali:", np.bincount(y_val))
    print("Distribuzione delle classi predette:", np.bincount(y_val_pred))

    # Valuta il modello con il parametro zero_division per evitare l'avviso
    print("Report di classificazione del set di validazione:")
    print(classification_report(y_val, y_val_pred, zero_division=1))

    accuracy = accuracy_score(y_val, y_val_pred)
    print(f"Accuratezza sul set di validazione: {accuracy:.4f}")

    return svm_model

def train_lightgbm(X_train, y_train, X_val, y_val, n_steps=10):
    """
    Addestra un modello di LightGBM sul set di addestramento
    e valuta le sue prestazioni sul set di validazione.

    Args:
        X_train (ndarray): Caratteristiche del set di addestramento.
        y_train (ndarray): Target del set di addestramento.
        X_val (ndarray): Caratteristiche del set di validazione.
        y_val (ndarray): Target del set di validazione.
        n_steps (int): Numero di passaggi per l'addestramento incrementale.

    Returns:
        lgb_model: Il modello di LightGBM addestrato.
    """
    # Inizializza il modello LightGBM
    lgb_model = lgb.LGBMClassifier(random_state=42)

    # Suddividi artificialmente il set di addestramento in step per simulare l'avanzamento
    data_size = len(X_train)
    step_size = data_size // n_steps

    for step in tqdm(range(n_steps), desc="Training Progress"):
        start = step * step_size
        end = (step + 1) * step_size if (step + 1) * step_size < data_size else data_size

        # Fit con un blocco incrementale dei dati
        lgb_model.fit(X_train[:end], y_train[:end])

    # Previsioni sul set di validazione
    y_val_pred = lgb_model.predict(X_val)

    # Controlla la distribuzione delle classi reali e predette
    print("Distribuzione delle classi reali:", np.bincount(y_val))
    print("Distribuzione delle classi predette:", np.bincount(y_val_pred))

    # Valuta il modello con il parametro zero_division per evitare l'avviso
    print("Report di classificazione del set di validazione:")
    print(classification_report(y_val, y_val_pred, zero_division=1))

    accuracy = accuracy_score(y_val, y_val_pred)
    print(f"Accuratezza sul set di validazione: {accuracy:.4f}")

    return lgb_model
