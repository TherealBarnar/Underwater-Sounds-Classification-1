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


def apply_smote_multiclass(X_train, y_train, k_neighbors=4):
    """
    Applica SMOTE al set di addestramento per bilanciare le classi in un contesto multiclasse.

    Args:
        X_train (ndarray): Caratteristiche del set di addestramento.
        y_train (ndarray): Target del set di addestramento.
        k_neighbors (int): Numero di vicini utilizzati per generare nuovi campioni.

    Returns:
        tuple: (X_train_resampled, y_train_resampled)
            - X_train_resampled (ndarray): Caratteristiche del set di addestramento dopo SMOTE.
            - y_train_resampled (ndarray): Target del set di addestramento dopo SMOTE.
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


def train_random_forest_multiclass(X_train, y_train, X_val, y_val, n_steps=10):
    """
    Addestra un modello di Random Forest sul set di addestramento
    e valuta le sue prestazioni sul set di validazione con cross-validation
    e tuning degli iperparametri.

    Args:
        X_train (ndarray): Caratteristiche del set di addestramento.
        y_train (ndarray): Target del set di addestramento.
        X_val (ndarray): Caratteristiche del set di validazione.
        y_val (ndarray): Target del set di validazione.
        n_steps (int): Numero di step per l'addestramento incrementale.

    Returns:
        best_model: Miglior modello Random Forest addestrato.
    """
    # Definisci la griglia di iperparametri
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'class_weight': ['balanced']  # Gestione automatica del bilanciamento delle classi
    }

    # Usa StratifiedKFold per mantenere le proporzioni delle classi in ogni suddivisione
    cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)  # Ridotto il numero di suddivisioni a 2

    # Inizializza il modello Random Forest
    rf_model = RandomForestClassifier(random_state=42)

    # Configura la ricerca degli iperparametri con cross-validation stratificata
    grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=cv, n_jobs=-1, verbose=2)

    # Suddividi artificialmente il set di addestramento in step per simulare l'avanzamento
    data_size = len(X_train)
    step_size = data_size // n_steps

    for step in tqdm(range(n_steps), desc="Training Progress"):
        start = step * step_size
        end = (step + 1) * step_size if (step + 1) * step_size < data_size else data_size

        # Adatta il modello sui dati attuali fino al passo corrente
        grid_search.fit(X_train[:end], y_train[:end])

    # Ottieni il miglior modello dal GridSearch
    best_model = grid_search.best_estimator_

    # Previsioni sul set di validazione
    y_val_pred = best_model.predict(X_val)

    # Controlla la distribuzione delle classi reali e predette
    print("Distribuzione delle classi reali:", np.bincount(y_val))
    print("Distribuzione delle classi predette:", np.bincount(y_val_pred))

    # Valuta il modello con il parametro zero_division=0 per evitare l'avviso di divisione per zero
    print("Report di classificazione del set di validazione:")
    print(classification_report(y_val, y_val_pred, zero_division=0))

    # Calcola e stampa l'accuratezza
    accuracy = accuracy_score(y_val, y_val_pred)
    print(f"Accuratezza sul set di validazione: {accuracy:.4f}")

    return best_model



