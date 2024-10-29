
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
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

def split_dataset(csv_file, train_size=0.8, val_size=0.1, test_size=0.1):
    # Carica il CSV
    df = pd.read_csv(csv_file)

    # Conta il numero di campioni prima del filtraggio
    initial_count = df.shape[0]

    # Filtra le subclassi con meno di 25 campioni
    subclass_counts = df['Subclass'].value_counts()
    valid_subclasses = subclass_counts[subclass_counts >= 25].index
    df = df[df['Subclass'].isin(valid_subclasses)]

    # Conta i campioni dopo il filtraggio e calcola i campioni rimossi
    filtered_count = df.shape[0]
    removed_count = initial_count - filtered_count
    print(f"Numero di campioni rimossi per subclassi con meno di 25 campioni: {removed_count}")

    # Imputazione dei valori NaN solo nelle colonne numeriche
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    imputer = SimpleImputer(strategy='mean')
    df[numeric_cols] = imputer.fit_transform(df[numeric_cols])

    # Separa le caratteristiche e il target
    X = df.drop(columns=['Class', 'Subclass', 'File Name'])
    y = df['Class']
    subclasses = df['Subclass']

    # Codifica le etichette "Class" in numeri interi
    class_encoder = LabelEncoder()
    y_encoded = class_encoder.fit_transform(y)

    # Primo split train-temp (80% train, 20% temp), stratificato sulle subclassi
    X_train, X_temp, y_train_encoded, y_temp, subclasses_train, subclasses_temp = train_test_split(
        X, y_encoded, subclasses, train_size=train_size, stratify=subclasses, random_state=42)

    # Secondo split temp in validation-test (10% ciascuno), stratificato sulle subclassi rimanenti
    X_val, X_test, y_val_encoded, y_test_encoded, subclasses_val, subclasses_test = train_test_split(
        X_temp, y_temp, subclasses_temp, test_size=0.5, stratify=subclasses_temp, random_state=42)

    # Stampa la distribuzione delle subclassi nei vari set
    print("Distribuzione delle subclassi nel set di addestramento:")
    print(subclasses_train.value_counts())
    print("Distribuzione delle subclassi nel set di validazione:")
    print(subclasses_val.value_counts())
    print("Distribuzione delle subclassi nel set di test:")
    print(subclasses_test.value_counts())

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