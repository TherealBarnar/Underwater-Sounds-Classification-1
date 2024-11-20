
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import lightgbm as lgb
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, accuracy_score
import time
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
import pandas as pd
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

def split_dataset(df, train_size=0.8, val_size=0.1, test_size=0.1):
    # Aggiungi una colonna "Parent" per il raggruppamento coerente
    df['Parent'] = df['File Name'].str.extract(r'^(.*?)(?=_seg)')

    # Filtra i dati per le classi Target e Non-Target
    df_binary = df[df['Class'].isin(['Target', 'Non-Target'])].copy()

    # Codifica le classi in numeri: Target (1) e Non-Target (0)
    df_binary['Class_encoded'] = (df_binary['Class'] == 'Target').astype(int)

    print(f"Dimensione totale: {df_binary.shape[0]} campioni")

    # Suddivisione rispettando i gruppi
    gss = GroupShuffleSplit(n_splits=1, train_size=train_size, random_state=42)
    groups = df_binary['Parent']
    train_idx, temp_idx = next(gss.split(df_binary, groups=groups))

    # Suddivisione del set temporaneo tra validation e test
    val_test_split = GroupShuffleSplit(n_splits=1, train_size=val_size / (val_size + test_size), random_state=42)
    val_idx, test_idx = next(val_test_split.split(df_binary.iloc[temp_idx], groups=df_binary.iloc[temp_idx]['Parent']))

    # Creazione dei set di addestramento, validazione e test
    X_train, X_val, X_test = df_binary.iloc[train_idx].copy(), df_binary.iloc[val_idx].copy(), df_binary.iloc[test_idx].copy()

    # Separazione delle etichette
    y_train = X_train['Class_encoded'].values
    y_val = X_val['Class_encoded'].values
    y_test = X_test['Class_encoded'].values

    # Rimozione delle colonne aggiuntive
    for dataset in [X_train, X_val, X_test]:
        dataset.drop(columns=['Class_encoded'], inplace=True)

    # Stampa distribuzioni per verifica
    total_samples = df_binary.shape[0]
    print(f"\nDimensione del set di addestramento: {X_train.shape[0]} campioni ({X_train.shape[0] / total_samples:.2%})")
    print(f"Dimensione del set di validazione: {X_val.shape[0]} campioni ({X_val.shape[0] / total_samples:.2%})")
    print(f"Dimensione del set di test: {X_test.shape[0]} campioni ({X_test.shape[0] / total_samples:.2%})\n")

    # Verifica delle distribuzioni delle classi
    print("Distribuzione delle classi nel set di addestramento:")
    print(X_train['Class'].value_counts())
    print("\nDistribuzione delle classi nel set di validazione:")
    print(X_val['Class'].value_counts())
    print("\nDistribuzione delle classi nel set di test:")
    print(X_test['Class'].value_counts())

    return X_train, X_val, X_test, y_train, y_val, y_test


def apply_smote(X_train, y_train, subclasses=None):
    """
    Funzione per bilanciare le classi usando SMOTE.
    Args:
    - X_train (pd.DataFrame): Le caratteristiche di addestramento.
    - y_train (pd.Series): I target di addestramento.
    - subclasses (list): Le sottoclassi che devono essere bilanciate.

    Returns:
    - X_train_resampled (pd.DataFrame): Il dataset delle caratteristiche bilanciato.
    - y_train_resampled (pd.Series): Il target bilanciato.
    """

    # Verifica delle classi presenti
    if subclasses is not None:
        y_train = y_train[y_train.isin(subclasses)]

    # Applicare SMOTE solo per le classi
    smote = SMOTE(sampling_strategy='auto')  # 'auto' significa bilanciare automaticamente le classi
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    return X_train_resampled, y_train_resampled


def train_random_forest(X_train, y_train, X_val, y_val, X_test, y_test):

    # Definisci il modello Random Forest
    rf_model = RandomForestClassifier(random_state=42, n_estimators=100, warm_start=True)

    # Imposta la progress bar con tqdm
    print("Inizio dell'addestramento del modello Random Forest...")
    for i in tqdm(range(1, rf_model.n_estimators + 1), desc="Addestramento Random Forest", unit="albero"):
        rf_model.set_params(n_estimators=i)  # Aggiunge un albero per iterazione
        rf_model.fit(X_train, y_train)

    # Previsioni sul set di validazione
    y_val_pred = rf_model.predict(X_val)
    print("Performance sul set di validazione:")
    print(classification_report(y_val, y_val_pred))
    print("Accuratezza sul set di validazione:", accuracy_score(y_val, y_val_pred))

    # Previsioni sul set di test
    y_test_pred = rf_model.predict(X_test)
    print("Performance sul set di test:")
    print(classification_report(y_test, y_test_pred))
    print("Accuratezza sul set di test:", accuracy_score(y_test, y_test_pred))

    return rf_model
def train_svm(X_train_resampled_array, y_train_resampled, X_val_imputed_array, y_val_encoded, X_test_imputed_array,
              y_test_encoded):

    # Crea un pipeline per normalizzare i dati e addestrare il modello
    pipeline = make_pipeline(StandardScaler(), SGDClassifier(loss='hinge', random_state=42, n_jobs=-1))

    print("Inizio addestramento del modello...")
    start_time = time.time()

    # Addestramento del modello
    pipeline.fit(X_train_resampled_array, y_train_resampled)

    end_time = time.time()
    print("Tempo di addestramento:", end_time - start_time)

    # Previsione sul set di validazione
    print("Inizio previsione sul set di validazione...")
    start_val_time = time.time()
    y_val_predictions = pipeline.predict(X_val_imputed_array)
    end_val_time = time.time()
    print("Tempo di previsione sul set di validazione:", end_val_time - start_val_time)

    # Valutazione delle prestazioni sul set di validazione
    print("Performance del set di validazione:")
    print("Accuracy:", accuracy_score(y_val_encoded, y_val_predictions))
    print(classification_report(y_val_encoded, y_val_predictions, target_names=["Non-Target", "Target"], digits=4))

    # Previsione sul set di test
    print("Inizio previsione sul set di test...")
    start_test_time = time.time()
    test_predictions = pipeline.predict(X_test_imputed_array)
    end_test_time = time.time()
    print("Tempo di previsione sul set di test:", end_test_time - start_test_time)

    # Valutazione delle prestazioni sul set di test
    print("Performance del set di test:")
    print("Accuracy:", accuracy_score(y_test_encoded, test_predictions))
    print(classification_report(y_test_encoded, test_predictions, target_names=["Non-Target", "Target"], digits=4))

    return pipeline
def train_lightgbm(X_train, y_train, X_val, y_val, X_test, y_test):
    # Rimuove valori NaN
    y_train = y_train[~pd.isna(y_train)]
    y_val = y_val[~pd.isna(y_val)]
    y_test = y_test[~pd.isna(y_test)]


    # Verifica e mappa le etichette
    unique_classes = np.unique(y_train)
    label_mapping = {label: i for i, label in enumerate(unique_classes)}

    # Mappatura con gestione di valori non trovati
    y_train_mapped = np.vectorize(lambda x: label_mapping.get(x, -1))(y_train)
    y_val_mapped = np.vectorize(lambda x: label_mapping.get(x, -1))(y_val)
    y_test_mapped = np.vectorize(lambda x: label_mapping.get(x, -1))(y_test)

    # Controllo per valori mappati a -1
    if np.any(y_train_mapped == -1) or np.any(y_val_mapped == -1) or np.any(y_test_mapped == -1):
        print("Attenzione: ci sono etichette non valide (None o non mappabili) nel set di dati.")
        print("Etichette non mappabili in y_train:", y_train[np.isin(y_train, unique_classes, invert=True)])
        print("Etichette non mappabili in y_val:", y_val[np.isin(y_val, unique_classes, invert=True)])
        print("Etichette non mappabili in y_test:", y_test[np.isin(y_test, unique_classes, invert=True)])
        return None

    # Crea un dataset LightGBM
    train_data = lgb.Dataset(X_train, label=y_train_mapped)
    val_data = lgb.Dataset(X_val, label=y_val_mapped, reference=train_data)

    # Imposta i parametri del modello
    params = {
        'objective': 'multiclass',
        'num_class': len(unique_classes),
        'metric': 'multi_logloss',
        'random_state': 42,
        'verbose': -1
    }

    # Addestra il modello
    lgb_model = lgb.train(params, train_data, valid_sets=[val_data], valid_names=['valid'], num_boost_round=1000)

    # Previsioni sul set di validazione
    y_val_pred = lgb_model.predict(X_val, num_iteration=lgb_model.best_iteration)
    y_val_pred_classes = np.argmax(y_val_pred, axis=1)

    # Valuta il modello sul set di validazione
    print("Report di classificazione del set di validazione:")
    print(classification_report(y_val_mapped, y_val_pred_classes, zero_division=1))

    accuracy_val = accuracy_score(y_val_mapped, y_val_pred_classes)
    print(f"Accuratezza sul set di validazione: {accuracy_val:.4f}")

    # Previsioni sul set di test
    y_test_pred = lgb_model.predict(X_test, num_iteration=lgb_model.best_iteration)
    y_test_pred_classes = np.argmax(y_test_pred, axis=1)

    # Valuta il modello sul set di test
    print("Report di classificazione del set di test:")
    print(classification_report(y_test_mapped, y_test_pred_classes, zero_division=1))

    accuracy_test = accuracy_score(y_test_mapped, y_test_pred_classes)
    print(f"Accuratezza sul set di test: {accuracy_test:.4f}")

    return lgb_model