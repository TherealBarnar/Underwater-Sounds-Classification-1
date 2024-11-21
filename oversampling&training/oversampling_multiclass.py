import pandas as pd
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import lightgbm as lgb
from sklearn.metrics import classification_report, log_loss, accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder

def split_dataset(df, train_size=0.8, val_size=0.1, test_size=0.1):

    # Codifica le etichette "Subclass" in numeri interi
    subclass_encoder = LabelEncoder()
    df['Subclass_encoded'] = subclass_encoder.fit_transform(df['Subclass'])

    # Suddivisione per subclass mantenendo coerenza di gruppi
    train_idx, val_idx, test_idx = [], [], []
    for subclass in df['Subclass'].unique():
        subclass_data = df[df['Subclass'] == subclass]

        # Utilizzo di GroupShuffleSplit per dividere ogni subclass rispettando la coesione dei gruppi
        gss = GroupShuffleSplit(n_splits=1, train_size=train_size, random_state=42)
        groups = subclass_data['Parent']
        subclass_train_idx, subclass_temp_idx = next(gss.split(subclass_data, groups=groups))

        # Suddivisione del set temporaneo tra validation e test
        val_test_split = GroupShuffleSplit(n_splits=1, train_size=val_size / (val_size + test_size), random_state=42)
        val_idx_local, test_idx_local = next(val_test_split.split(subclass_data.iloc[subclass_temp_idx],
                                                                  groups=subclass_data.iloc[subclass_temp_idx][
                                                                      'Parent']))

        # Aggiusta gli indici per il dataframe originale
        train_idx.extend(subclass_data.index[subclass_train_idx].tolist())
        val_idx.extend(subclass_data.index[subclass_temp_idx[val_idx_local]].tolist())
        test_idx.extend(subclass_data.index[subclass_temp_idx[test_idx_local]].tolist())

    # Creazione dei set di addestramento, validazione e test
    X_train, X_val, X_test = df.loc[train_idx].copy(), df.loc[val_idx].copy(), df.loc[
        test_idx].copy()

    # Ordina i set per prefisso e numero di segmento
    for dataset in [X_train, X_val, X_test]:
        dataset.loc[:, 'File_Prefix'] = dataset['File Name'].str.extract(r'(^[a-f0-9\-]+)')
        dataset.loc[:, 'Segment_Number'] = dataset['File Name'].str.extract(r'_seg(\d+)').astype(int)
        dataset.sort_values(by=['File_Prefix', 'Segment_Number'], inplace=True)
        dataset.drop(columns=['File_Prefix', 'Segment_Number'], inplace=True)

    # Separazione delle etichette
    y_train = X_train['Subclass_encoded'].values
    y_val = X_val['Subclass_encoded'].values
    y_test = X_test['Subclass_encoded'].values

    # Rimozione delle colonne aggiuntive dai dataframe di output
    X_train = X_train.drop(columns=['Subclass_encoded'])
    X_val = X_val.drop(columns=['Subclass_encoded'])
    X_test = X_test.drop(columns=['Subclass_encoded'])

    # Stampa distribuzioni per verifica
    total_samples = df.shape[0]
    print(
        f"\nDimensione del set di addestramento: {X_train.shape[0]} campioni ({X_train.shape[0] / total_samples:.2%})")
    print(f"Dimensione del set di validazione: {X_val.shape[0]} campioni ({X_val.shape[0] / total_samples:.2%})")
    print(f"Dimensione del set di test: {X_test.shape[0]} campioni ({X_test.shape[0] / total_samples:.2%})\n")

    # Verifica delle distribuzioni delle subclass
    print("Distribuzione delle subclass nel set di addestramento:")
    print(X_train['Subclass'].value_counts())
    print("\nDistribuzione delle subclass nel set di validazione:")
    print(X_val['Subclass'].value_counts())
    print("\nDistribuzione delle subclass nel set di test:")
    print(X_test['Subclass'].value_counts())

    return X_train, X_val, X_test, y_train, y_val, y_test, subclass_encoder

def apply_smote_multiclass(X_train, y_train, k_neighbors=1):
    """
    Applica SMOTE al set di addestramento per bilanciare le classi in un contesto multiclasse.
    Se una classe ha solo 1 campione, quella classe verrà rimossa.
    """
    # Converti y_train in una pandas Series per utilizzare .value_counts() e .isin()
    y_train_series = pd.Series(y_train)

    # Verifica la distribuzione delle classi
    class_counts = y_train_series.value_counts()

    # Rimuovi le classi con solo 1 campione
    to_remove = class_counts[class_counts == 1].index
    if len(to_remove) > 0:
        print(f"Attenzione: Le seguenti classi hanno solo 1 campione e saranno rimosse: {list(to_remove)}")
        mask = ~y_train_series.isin(to_remove)
        X_train = X_train[mask]
        y_train_series = y_train_series[mask]

    y_train = y_train_series.values  # Aggiorna y_train dopo il filtro

    # Elimina le colonne categoriche prima di SMOTE
    X_train_num = X_train.select_dtypes(include=[np.number])

    # Verifica il numero minimo di campioni per la classe meno rappresentata
    class_counts = pd.Series(y_train).value_counts()
    min_samples = class_counts.min()

    # Assicurati che k_neighbors sia inferiore o uguale al numero di campioni minimi meno 1
    if k_neighbors > min_samples - 1:
        print(
            f"Attenzione: Il numero di vicini ({k_neighbors}) è maggiore del numero di campioni nella classe meno rappresentata ({min_samples}). Ridurre k_neighbors."
        )
        k_neighbors = min_samples - 1  # Riduci k_neighbors al massimo possibile

    # Crea l'istanza di SMOTE per il caso multiclasse
    smote = SMOTE(k_neighbors=k_neighbors, random_state=42)

    # Applica SMOTE al set di addestramento solo sui dati numerici
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_num, y_train)

    # Verifica la distribuzione delle classi dopo SMOTE
    print(f"Distribuzione delle classi nel set di training dopo SMOTE:\n{pd.Series(y_train_resampled).value_counts()}")

    return X_train_resampled, y_train_resampled

"""
def train_random_forest_multiclass_0(X_train, y_train, X_val, y_val, X_test, y_test):

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
"""

def train_random_forest_multiclass(X_train, y_train, X_val, y_val, X_test, y_test):
    # Rimuovi colonne non necessarie
    X_train = X_train.drop(columns=['Class', 'Parent', 'Subclass', 'File Name'], errors='ignore')
    X_val = X_val.drop(columns=['Class', 'Parent', 'Subclass', 'File Name'], errors='ignore')
    X_test = X_test.drop(columns=['Class', 'Parent', 'Subclass', 'File Name'], errors='ignore')

    # Trova tutte le classi presenti nei set
    all_classes = np.unique(np.concatenate((y_train, y_val, y_test)))

    # Impostazioni per la Grid Search
    param_grid = {
        'n_estimators': [100, 150, 200],
        'max_depth': [5, 7, 9],
        'min_samples_split': [10, 15],
        'min_samples_leaf': [4, 5]
    }

    model = RandomForestClassifier(max_features="sqrt", random_state=42)
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_

    # Valutazione su Validation Set
    y_val_pred = best_model.predict(X_val)
    y_val_proba = best_model.predict_proba(X_val)
    accuracy_val = accuracy_score(y_val, y_val_pred)
    logloss_val = log_loss(y_val, y_val_proba, labels=np.unique(np.concatenate((y_val, y_val_pred))))
    print(f"Accuratezza sul Validation Set: {accuracy_val:.4f}")
    print(f"Log Loss sul Validation Set: {logloss_val:.4f}")

    # Report di classificazione per il Validation Set
    print("\n=== Report di Classificazione - Validation Set ===")
    print(classification_report(y_val, y_val_pred, labels=np.unique(np.concatenate((y_val, y_val_pred))),
                                zero_division=0))

    # Valutazione su Test Set
    y_test_pred = best_model.predict(X_test)
    y_test_proba = best_model.predict_proba(X_test)
    accuracy_test = accuracy_score(y_test, y_test_pred)
    logloss_test = log_loss(y_test, y_test_proba, labels=np.unique(np.concatenate((y_test, y_test_pred))))
    print(f"Accuratezza sul Test Set: {accuracy_test:.4f}")
    print(f"Log Loss sul Test Set: {logloss_test:.4f}")

    # Report di classificazione per il Test Set
    print("\n=== Report di Classificazione - Test Set ===")
    print(classification_report(y_test, y_test_pred, labels=np.unique(np.concatenate((y_test, y_test_pred))),
                                zero_division=0))

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
    # Rimuove valori NaN da y e applica la maschera a X
    na_mask_train = ~pd.isna(y_train)
    X_train = X_train[na_mask_train]
    y_train = y_train[na_mask_train]
    print(f"Dimensioni dopo NaN nel set di addestramento: X_train: {X_train.shape}, y_train: {y_train.shape}")

    na_mask_val = ~pd.isna(y_val)
    X_val = X_val[na_mask_val]
    y_val = y_val[na_mask_val]
    print(f"Dimensioni dopo NaN nel set di validazione: X_val: {X_val.shape}, y_val: {y_val.shape}")

    na_mask_test = ~pd.isna(y_test)
    X_test = X_test[na_mask_test]
    y_test = y_test[na_mask_test]
    print(f"Dimensioni dopo NaN nel set di test: X_test: {X_test.shape}, y_test: {y_test.shape}")

    # Controlla le classi uniche
    print("Classi nel set di addestramento:", np.unique(y_train))
    print("Classi nel set di validazione:", np.unique(y_val))
    print("Classi nel set di test:", np.unique(y_test))

    # Mappa le etichette
    label_encoder = LabelEncoder()
    y_train_mapped = label_encoder.fit_transform(y_train)
    y_val_mapped = label_encoder.transform(y_val)
    y_test_mapped = label_encoder.transform(y_test)

    # Crea un dataset LightGBM
    train_data = lgb.Dataset(X_train, label=y_train_mapped)
    val_data = lgb.Dataset(X_val, label=y_val_mapped, reference=train_data)

    # Imposta i parametri del modello
    params = {
        'objective': 'multiclass',
        'num_class': len(np.unique(y_train)),  # Dovrebbe ora essere il numero corretto di classi
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


def rf_plot_confusion_matrices(model, X_val, y_val_encoded, X_test, y_test_encoded):

    # Predizioni del modello
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)

    # Genera le matrici di confusione
    cm_val = confusion_matrix(y_val_encoded, y_val_pred, labels=np.unique(y_val_encoded))
    cm_test = confusion_matrix(y_test_encoded, y_test_pred, labels=np.unique(y_test_encoded))

    # Genera le rappresentazioni grafiche
    disp_val = ConfusionMatrixDisplay(confusion_matrix=cm_val, display_labels=np.unique(y_val_encoded))
    disp_test = ConfusionMatrixDisplay(confusion_matrix=cm_test, display_labels=np.unique(y_test_encoded))

    # Crea una figura con due sotto-grafici
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Mostra le matrici di confusione
    disp_val.plot(cmap='Blues', ax=axes[0])
    axes[0].set_title("Random Forest - Matrice di Confusione (Validazione)")

    disp_test.plot(cmap='Blues', ax=axes[1])
    axes[1].set_title("Random Forest - Matrice di Confusione (Test)")

    # Rimuove la griglia
    for ax in axes:
        ax.grid(False)

    # Mostra le figure
    plt.tight_layout()
    plt.show()


def svm_plot_confusion_matrices(model, X_val, y_val_encoded, X_test, y_test_encoded):

    # Predizioni del modello
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)

    # Genera le matrici di confusione
    cm_val = confusion_matrix(y_val_encoded, y_val_pred, labels=np.unique(y_val_encoded))
    cm_test = confusion_matrix(y_test_encoded, y_test_pred, labels=np.unique(y_test_encoded))

    # Genera le rappresentazioni grafiche
    disp_val = ConfusionMatrixDisplay(confusion_matrix=cm_val, display_labels=np.unique(y_val_encoded))
    disp_test = ConfusionMatrixDisplay(confusion_matrix=cm_test, display_labels=np.unique(y_test_encoded))

    # Crea una figura con due sotto-grafici
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Mostra le matrici di confusione
    disp_val.plot(cmap='Blues', ax=axes[0])
    axes[0].set_title("SVM - Matrice di Confusione (Validazione)")

    disp_test.plot(cmap='Blues', ax=axes[1])
    axes[1].set_title("SVM - Matrice di Confusione (Test)")

    # Rimuove la griglia
    for ax in axes:
        ax.grid(False)

    # Mostra le figure
    plt.tight_layout()
    plt.show()

def lightgbm_plot_confusion_matrices(model, X_val, y_val_encoded, X_test, y_test_encoded):

    y_val_pred_proba = model.predict(X_val)
    y_val_pred = np.argmax(y_val_pred_proba, axis=1)

    y_test_pred_proba = model.predict(X_test)
    y_test_pred = np.argmax(y_test_pred_proba, axis=1)

    # Genera le matrici di confusione
    cm_val = confusion_matrix(y_val_encoded, y_val_pred, labels=np.unique(y_val_encoded))
    cm_test = confusion_matrix(y_test_encoded, y_test_pred, labels=np.unique(y_test_encoded))

    # Genera le rappresentazioni grafiche
    disp_val = ConfusionMatrixDisplay(confusion_matrix=cm_val, display_labels=np.unique(y_val_encoded))
    disp_test = ConfusionMatrixDisplay(confusion_matrix=cm_test, display_labels=np.unique(y_test_encoded))

    # Crea una figura con due sotto-grafici
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Mostra le matrici di confusione
    disp_val.plot(cmap='Blues', ax=axes[0])
    axes[0].set_title("LightGBM - Matrice di Confusione (Validazione)")

    disp_test.plot(cmap='Blues', ax=axes[1])
    axes[1].set_title("LightGBM - Matrice di Confusione (Test)")

    # Rimuove la griglia
    for ax in axes:
        ax.grid(False)

    # Mostra le figure
    plt.tight_layout()
    plt.show()
