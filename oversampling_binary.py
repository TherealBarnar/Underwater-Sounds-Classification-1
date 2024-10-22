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
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import time
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier


def split_dataset(csv_file, train_size=0.8, val_size=0.1, test_size=0.1):
    # Carica il CSV
    df = pd.read_csv(csv_file)

    # Estrai il nome base del file audio
    df['BaseFileName'] = df['File Name'].str.split('_seg').str[0]

    # Raggruppa per 'BaseFileName'
    file_names = df['BaseFileName'].unique()

    # Suddividi i nomi dei file in train, validation e test usando le proporzioni fornite
    train_file_names, temp_file_names = train_test_split(file_names, test_size=(val_size + test_size), random_state=42)
    val_file_names, test_file_names = train_test_split(temp_file_names, test_size=(test_size / (val_size + test_size)), random_state=42)

    # Filtra i dati per ciascun set
    train_df = df[df['BaseFileName'].isin(train_file_names)]
    val_df = df[df['BaseFileName'].isin(val_file_names)]
    test_df = df[df['BaseFileName'].isin(test_file_names)]

    # Rimuovi le colonne non necessarie
    train_df = train_df.drop(columns=['BaseFileName', 'File Name'])
    val_df = val_df.drop(columns=['BaseFileName', 'File Name'])
    test_df = test_df.drop(columns=['BaseFileName', 'File Name'])

    # Imputazione dei valori NaN solo nelle colonne numeriche
    numeric_cols = train_df.select_dtypes(include=['float64', 'int64']).columns
    imputer = SimpleImputer(strategy='mean')
    train_df[numeric_cols] = imputer.fit_transform(train_df[numeric_cols])
    val_df[numeric_cols] = imputer.transform(val_df[numeric_cols])
    test_df[numeric_cols] = imputer.transform(test_df[numeric_cols])

    # Separa le caratteristiche e il target
    X_train = train_df.drop(columns=['Class', 'Subclass'])
    y_train = train_df['Class']
    X_val = val_df.drop(columns=['Class', 'Subclass'])
    y_val = val_df['Class']
    X_test = test_df.drop(columns=['Class', 'Subclass'])
    y_test = test_df['Class']
    # Codifica le etichette "Class" in numeri interi
    class_encoder = LabelEncoder()
    y_train_encoded = class_encoder.fit_transform(y_train)
    y_val_encoded = class_encoder.transform(y_val)
    y_test_encoded = class_encoder.transform(y_test)

    # Mantieni le subclassi per l'addestramento
    subclasses_train = train_df['Subclass']

    # Stampa la distribuzione delle classi nel set di addestramento
    print("Distribuzione delle classi nel set di addestramento:")
    print(y_train.value_counts())

    # Stampa la distribuzione delle subclassi nel set di addestramento
    print("Distribuzione delle subclassi nel set di addestramento:")
    print(subclasses_train.value_counts())

    # Controlla le dimensioni
    print("Dimensioni del set di addestramento:", X_train.shape, y_train_encoded.shape)
    print("Dimensioni del set di validazione:", X_val.shape, y_val_encoded.shape)
    print("Dimensioni del set di test:", X_test.shape, y_test_encoded.shape)

    return X_train, X_val, X_test, y_train_encoded, y_val_encoded, y_test_encoded, subclasses_train, class_encoder


def apply_smote(X_train, y_train, subclasses_train):
    # Imputazione dei valori NaN
    imputer = SimpleImputer(strategy='mean')
    X_train_imputed = imputer.fit_transform(X_train)

    # Separiamo i campioni tra Target (1) e Non-Target (0)
    X_target = X_train_imputed[y_train == 1]
    subclasses_target = subclasses_train[y_train == 1]

    X_non_target = X_train_imputed[y_train == 0]
    subclasses_non_target = subclasses_train[y_train == 0]

    # Rimuoviamo le subclassi con un solo campione
    subclass_counts_target = pd.Series(subclasses_target).value_counts()
    valid_subclasses_target = subclass_counts_target[subclass_counts_target > 1].index
    X_target = X_target[np.isin(subclasses_target, valid_subclasses_target)]
    subclasses_target = subclasses_target[np.isin(subclasses_target, valid_subclasses_target)]

    subclass_counts_non_target = pd.Series(subclasses_non_target).value_counts()
    valid_subclasses_non_target = subclass_counts_non_target[subclass_counts_non_target > 1].index
    X_non_target = X_non_target[np.isin(subclasses_non_target, valid_subclasses_non_target)]
    subclasses_non_target = subclasses_non_target[np.isin(subclasses_non_target, valid_subclasses_non_target)]

    # Crea l'istanza di SMOTE
    smote = SMOTE(random_state=42, k_neighbors=1)

    # SMOTE tra le subclass di Target
    print("SMOTE tra le subclass di Target:")
    if len(X_target) > 1:
        X_target_resampled, subclasses_target_resampled = smote.fit_resample(X_target, subclasses_target)
        print("Distribuzione subclassi di Target dopo SMOTE:")
        print(pd.Series(subclasses_target_resampled).value_counts())
    else:
        print("Non ci sono abbastanza campioni nella subclass di Target.")
        X_target_resampled = np.empty((0, X_target.shape[1]))
        subclasses_target_resampled = np.array([])

    # SMOTE tra le subclass di Non-Target
    print("SMOTE tra le subclass di Non-Target:")
    if len(X_non_target) > 1:
        X_non_target_resampled, subclasses_non_target_resampled = smote.fit_resample(X_non_target, subclasses_non_target)
        print("Distribuzione subclassi di Non-Target dopo SMOTE:")
        print(pd.Series(subclasses_non_target_resampled).value_counts())
    else:
        print("Non ci sono abbastanza campioni nella subclass di Non-Target.")
        X_non_target_resampled = np.empty((0, X_non_target.shape[1]))
        subclasses_non_target_resampled = np.array([])

    # Combiniamo i risultati
    if X_target_resampled.shape[0] > 0 or X_non_target_resampled.shape[0] > 0:
        X_resampled_combined = np.vstack([X_target_resampled, X_non_target_resampled])
        y_resampled_combined = np.hstack([np.ones(len(X_target_resampled)), np.zeros(len(X_non_target_resampled))])

        # SMOTE globale finale tra Target e Non-Target
        print("SMOTE globale finale tra Target e Non-Target:")
        if len(y_resampled_combined) > 1:
            X_train_final_resampled, y_train_final_resampled = smote.fit_resample(X_resampled_combined, y_resampled_combined)
            print("Distribuzione delle classi dopo SMOTE globale finale:")
            print(pd.Series(y_train_final_resampled).value_counts())
        else:
            raise ValueError("Nessun campione disponibile per lo SMOTE globale finale tra Target e Non-Target.")

        # Stampa delle dimensioni finali
        print("Dimensioni finali del set di addestramento:", X_train_final_resampled.shape, y_train_final_resampled.shape)

        return X_train_final_resampled, y_train_final_resampled
    else:
        raise ValueError("Nessun campione disponibile dopo SMOTE tra le subclassi.")




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
def train_lightgbm(X_train, y_train, X_val, y_val, X_test, y_test, n_steps=10):
    """
    Addestra un modello di LightGBM sul set di addestramento
    e valuta le sue prestazioni sul set di validazione e test.

    Args:
        X_train (ndarray): Caratteristiche del set di addestramento.
        y_train (ndarray): Target del set di addestramento.
        X_val (ndarray): Caratteristiche del set di validazione.
        y_val (ndarray): Target del set di validazione.
        X_test (ndarray): Caratteristiche del set di test.
        y_test (ndarray): Target del set di test.
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
    print("Distribuzione delle classi reali (validazione):", np.bincount(y_val))
    print("Distribuzione delle classi predette (validazione):", np.bincount(y_val_pred))

    # Valuta il modello sul set di validazione
    print("Report di classificazione del set di validazione:")
    print(classification_report(y_val, y_val_pred, zero_division=1))

    accuracy_val = accuracy_score(y_val, y_val_pred)
    print(f"Accuratezza sul set di validazione: {accuracy_val:.4f}")

    # Previsioni sul set di test
    y_test_pred = lgb_model.predict(X_test)

    # Controlla la distribuzione delle classi reali e predette
    print("Distribuzione delle classi reali (test):", np.bincount(y_test))
    print("Distribuzione delle classi predette (test):", np.bincount(y_test_pred))

    # Valuta il modello sul set di test
    print("Report di classificazione del set di test:")
    print(classification_report(y_test, y_test_pred, zero_division=1))

    accuracy_test = accuracy_score(y_test, y_test_pred)
    print(f"Accuratezza sul set di test: {accuracy_test:.4f}")

    return lgb_model