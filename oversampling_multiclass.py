import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from collections import Counter
from sklearn.svm import SVC  # Importa SVC per il modello SVM
import lightgbm as lgb  # Importa LightGBM


# Funzione per suddividere il dataset
def split_dataset(csv_file, train_size=0.8, val_size=0.1, test_size=0.1):
    """
    Carica un dataset CSV, lo divide in set di addestramento, validazione e test,
    e restituisce le caratteristiche e i target separati.
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


def apply_smote_multiclass(X_train, y_train, k_neighbors=1):
    """
    Applica SMOTE al set di addestramento per bilanciare le classi in un contesto multiclasse.
    """
    # Verifica la distribuzione delle classi
    class_counts = pd.Series(y_train).value_counts()

    # Verifica il numero minimo di campioni per la classe meno rappresentata
    min_samples = class_counts.min()

    # Assicurati che k_neighbors sia inferiore o uguale al numero di campioni minimi
    if k_neighbors > min_samples - 1:
        print(
            f"Attenzione: Il numero di vicini ({k_neighbors}) è maggiore del numero di campioni nella classe meno rappresentata ({min_samples}). Ridurre k_neighbors.")
        k_neighbors = min_samples - 1  # Riduci k_neighbors al massimo possibile

    # Crea l'istanza di SMOTE per il caso multiclasse
    smote = SMOTE(k_neighbors=k_neighbors, random_state=42)

    # Applica SMOTE al set di addestramento
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    # Verifica la distribuzione delle classi dopo SMOTE
    print(f"Distribuzione delle classi nel set di training dopo SMOTE:\n{pd.Series(y_train_resampled).value_counts()}")

    return X_train_resampled, y_train_resampled


def train_random_forest_multiclass(X_train, y_train, X_val, y_val, X_test, y_test, n_steps=10, min_samples_threshold=5):
    """
    Addestra un modello di Random Forest sul set di addestramento
    e valuta le sue prestazioni sul set di validazione e sul set di test
    con cross-validation e tuning degli iperparametri.

    Ignora le sottoclassi con meno di `min_samples_threshold` campioni.
    """
    # Verifica la distribuzione delle classi
    class_counts = Counter(y_train)
    print("Distribuzione delle classi nel set di addestramento:", class_counts)

    # Filtra le sottoclassi con campioni insufficienti
    valid_classes = [cls for cls, count in class_counts.items() if count >= min_samples_threshold]
    print(f"Sottoclassi valide (con almeno {min_samples_threshold} campioni): {valid_classes}")

    # Filtra il dataset
    mask = np.isin(y_train, valid_classes)
    X_train_filtered = X_train[mask]
    y_train_filtered = y_train[mask]

    # Verifica la distribuzione delle classi dopo il filtro
    filtered_class_counts = Counter(y_train_filtered)
    print("Distribuzione delle classi nel set di addestramento filtrato:", filtered_class_counts)

    # Determina il numero di suddivisioni adatto
    min_class_count = min(filtered_class_counts.values())
    num_splits = 2

    cv = StratifiedKFold(n_splits=num_splits, shuffle=True, random_state=42)

    # Definisci la griglia di iperparametri
    param_grid = {
        'n_estimators': [100],
        'max_depth': [None],
        'min_samples_split': [2],
        'min_samples_leaf': [1],
        'max_features': ['sqrt'],
        'class_weight': ['balanced']
    }

    # Inizializza il modello Random Forest
    rf_model = RandomForestClassifier(random_state=42)

    # Configura la ricerca degli iperparametri con cross-validation stratificata
    grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=cv, n_jobs=2, verbose=1,
                               scoring='accuracy')

    # Suddividi artificialmente il set di addestramento in step per simulare l'avanzamento
    data_size = len(X_train_filtered)
    step_size = data_size // n_steps

    for step in tqdm(range(n_steps), desc="Training Progress"):
        start = step * step_size
        end = (step + 1) * step_size if (step + 1) * step_size < data_size else data_size

        # Adatta il modello sui dati attuali fino al passo corrente
        print(f"Training on data from index {start} to {end}")
        grid_search.fit(X_train_filtered[:end], y_train_filtered[:end])
        print(f"Step {step + 1} completed")

    # Ottieni il miglior modello dal GridSearch
    best_model = grid_search.best_estimator_

    # Previsioni sul set di validazione
    y_val_pred = best_model.predict(X_val)

    # Controlla la distribuzione delle classi reali e predette sul set di validazione
    print("Distribuzione delle classi reali nel set di validazione:", np.bincount(y_val))
    print("Distribuzione delle classi predette nel set di validazione:", np.bincount(y_val_pred))

    # Valuta il modello sul set di validazione
    print("Report di classificazione del set di validazione:")
    print(classification_report(y_val, y_val_pred, zero_division=0))

    # Calcola e stampa l'accuratezza sul set di validazione
    accuracy_val = accuracy_score(y_val, y_val_pred)
    print(f"Accuratezza sul set di validazione: {accuracy_val:.4f}")

    # Previsioni sul set di test
    y_test_pred = best_model.predict(X_test)

    # Controlla la distribuzione delle classi reali e predette sul set di test
    print("Distribuzione delle classi reali nel set di test:", np.bincount(y_test))
    print("Distribuzione delle classi predette nel set di test:", np.bincount(y_test_pred))

    # Valuta il modello sul set di test
    print("Report di classificazione del set di test:")
    print(classification_report(y_test, y_test_pred, zero_division=0))

    # Calcola e stampa l'accuratezza sul set di test
    accuracy_test = accuracy_score(y_test, y_test_pred)
    print(f"Accuratezza sul set di test: {accuracy_test:.4f}")

    # Valuta l'importanza delle caratteristiche
    feature_importances = best_model.feature_importances_
    print("Importanza delle caratteristiche:", feature_importances)

    return best_model


def train_svm_multiclass(X_train, y_train, X_val, y_val, X_test, y_test):
    """
    Addestra un modello di Support Vector Machine (SVM) sul set di addestramento
    e valuta le sue prestazioni sul set di validazione e sul set di test.
    """
    # Inizializza il modello SVM
    svm_model = SVC(random_state=42)

    # Addestra il modello
    svm_model.fit(X_train, y_train)

    # Previsioni sul set di validazione
    y_val_pred = svm_model.predict(X_val)

    # Controlla la distribuzione delle classi reali e predette nel set di validazione
    print("Distribuzione delle classi reali nel set di validazione:", np.bincount(y_val))
    print("Distribuzione delle classi predette nel set di validazione:", np.bincount(y_val_pred))

    # Valuta il modello con il parametro zero_division=0 per evitare l'avviso di divisione per zero
    print("Report di classificazione del set di validazione:")
    print(classification_report(y_val, y_val_pred, zero_division=0))

    # Calcola e stampa l'accuratezza sul set di validazione
    accuracy_val = accuracy_score(y_val, y_val_pred)
    print(f"Accuratezza sul set di validazione: {accuracy_val:.4f}")

    # Previsioni sul set di test
    y_test_pred = svm_model.predict(X_test)

    # Controlla la distribuzione delle classi reali e predette nel set di test
    print("Distribuzione delle classi reali nel set di test:", np.bincount(y_test))
    print("Distribuzione delle classi predette nel set di test:", np.bincount(y_test_pred))

    # Valuta il modello sul set di test
    print("Report di classificazione del set di test:")
    print(classification_report(y_test, y_test_pred, zero_division=0))

    # Calcola e stampa l'accuratezza sul set di test
    accuracy_test = accuracy_score(y_test, y_test_pred)
    print(f"Accuratezza sul set di test: {accuracy_test:.4f}")

    return svm_model


def train_lightgbm(X_train, y_train, X_val, y_val, X_test, y_test):
    """
    Addestra un modello LightGBM sul set di addestramento
    e valuta le sue prestazioni sul set di validazione e sul set di test.
    """
    # Crea un dataset LightGBM
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    # Imposta i parametri del modello
    params = {
        'objective': 'multiclass',
        'num_class': len(np.unique(y_train)),
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
    print(classification_report(y_val, y_val_pred_classes, zero_division=1))

    accuracy_val = accuracy_score(y_val, y_val_pred_classes)
    print(f"Accuratezza sul set di validazione: {accuracy_val:.4f}")

    # Previsioni sul set di test
    y_test_pred = lgb_model.predict(X_test, num_iteration=lgb_model.best_iteration)
    y_test_pred_classes = np.argmax(y_test_pred, axis=1)

    # Valuta il modello sul set di test
    print("Report di classificazione del set di test:")
    print(classification_report(y_test, y_test_pred_classes, zero_division=1))

    accuracy_test = accuracy_score(y_test, y_test_pred_classes)
    print(f"Accuratezza sul set di test: {accuracy_test:.4f}")

    return lgb_model
