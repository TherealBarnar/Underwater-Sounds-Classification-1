from imblearn.over_sampling import SMOTE
from onedal.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, log_loss, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import GroupShuffleSplit, GridSearchCV
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.impute import SimpleImputer
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import lightgbm as lgb

def split_dataset(df, train_size=0.8, val_size=0.1, test_size=0.1):

    # Gestione dei NaN con SimpleImputer
    imputer = SimpleImputer(strategy='mean')
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    df[numeric_columns] = imputer.fit_transform(df[numeric_columns])

    # Aggiungi colonne per ordinamento
    df['File_Prefix'] = df['File Name'].str.extract(r'(^[a-f0-9\-]+)')
    df['Segment_Number'] = df['File Name'].str.extract(r'_seg(\d+)').astype(int)

    # Suddivisione per subclass mantenendo la coerenza dei gruppi
    train_idx, val_idx, test_idx = [], [], []
    for subclass in df['Subclass'].unique():
        subclass_data = df[df['Subclass'] == subclass]

        # Primo split: 80% train, 20% temporaneo (validation + test)
        gss = GroupShuffleSplit(n_splits=1, train_size=train_size, random_state=42)
        groups = subclass_data['Parent']
        train_local_idx, temp_local_idx = next(gss.split(subclass_data, groups=groups))

        # Split temporaneo: 50% validation, 50% test
        temp_data = subclass_data.iloc[temp_local_idx]
        gss_temp = GroupShuffleSplit(n_splits=1, train_size=val_size / (val_size + test_size), random_state=42)
        val_local_idx, test_local_idx = next(gss_temp.split(temp_data, groups=temp_data['Parent']))

        # Aggiungi gli indici al dataset globale
        train_idx.extend(subclass_data.index[train_local_idx])
        val_idx.extend(temp_data.index[val_local_idx])
        test_idx.extend(temp_data.index[test_local_idx])

    # Creazione dei set di addestramento, validazione e test
    X_train, X_val, X_test = df.loc[train_idx].copy(), df.loc[val_idx].copy(), df.loc[test_idx].copy()

    # Ordina i set per prefisso e numero di segmento
    for dataset in [X_train, X_val, X_test]:
        dataset.sort_values(by=['File_Prefix', 'Segment_Number'], inplace=True)

    # Separazione delle etichette
    # Separazione delle etichette
    y_train = X_train['Class'].values
    y_val = X_val['Class'].values
    y_test = X_test['Class'].values

    # Encoder per le etichette 'Class'
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_val_encoded = le.transform(y_val)
    y_test_encoded = le.transform(y_test)

    # Mantieni o rimuovi colonne di ordinamento (opzionale)
    columns_to_drop = ['File_Prefix', 'Segment_Number']
    X_train.drop(columns=columns_to_drop, inplace=True)
    X_val.drop(columns=columns_to_drop, inplace=True)
    X_test.drop(columns=columns_to_drop, inplace=True)

    # Report della distribuzione
    total_samples = len(df)
    print(f"\nDimensione del set di addestramento: {len(X_train)} campioni ({len(X_train) / total_samples:.2%})")
    print(f"Dimensione del set di validazione: {len(X_val)} campioni ({len(X_val) / total_samples:.2%})")
    print(f"Dimensione del set di test: {len(X_test)} campioni ({len(X_test) / total_samples:.2%})\n")

    # Verifica delle distribuzioni delle subclass
    print("Distribuzione delle subclass nel set di addestramento:")
    print(X_train['Class'].value_counts())
    print("\nDistribuzione delle subclass nel set di validazione:")
    print(X_val['Class'].value_counts())
    print("\nDistribuzione delle subclass nel set di test:")
    print(X_test['Class'].value_counts())


    return X_train, X_val, X_test, y_train_encoded, y_val_encoded, y_test_encoded

def apply_smote(X_train, y_train, k_neighbors=1):

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

def train_random_forest(X_train, y_train, X_val, y_val, X_test, y_test):

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

def train_svm(X_train, y_train, X_val, y_val, X_test, y_test):
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()

    # Codifica le etichette se non sono numeriche
    if y_train.dtype == 'O':
        y_train = le.fit_transform(y_train)
        y_val = le.transform(y_val)
        y_test = le.transform(y_test)

    svm_model = SVC(random_state=42)
    svm_model.fit(X_train, y_train)

    y_val_pred = svm_model.predict(X_val)
    y_test_pred = svm_model.predict(X_test)

    print("Distribuzione delle classi reali nel set di validazione:", np.bincount(y_val))
    print("Distribuzione delle classi predette nel set di validazione:", np.bincount(y_val_pred))

    print("Report di classificazione del set di validazione:")
    print(classification_report(y_val, y_val_pred, zero_division=0))

    accuracy_val = accuracy_score(y_val, y_val_pred)
    print(f"Accuratezza sul set di validazione: {accuracy_val:.4f}")

    print("Distribuzione delle classi reali nel set di test:", np.bincount(y_test))
    print("Distribuzione delle classi predette nel set di test:", np.bincount(y_test_pred))

    print("Report di classificazione del set di test:")
    print(classification_report(y_test, y_test_pred, zero_division=0))

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
    disp_val.plot(cmap='Greens', ax=axes[0])
    axes[0].set_title("Random Forest - Matrice di Confusione (Validazione)")

    disp_test.plot(cmap='Greens', ax=axes[1])
    axes[1].set_title("Random Forest - Matrice di Confusione (Test)")

    # Rimuove la griglia
    for ax in axes:
        ax.grid(False)

    # Mostra le figure
    plt.tight_layout()
    plt.show()

def svm_plot_confusion_matrices(model, X_val, y_val, X_test, y_test):
    """
    Genera e visualizza le matrici di confusione per i dati di validazione e test.

    Parametri:
        - model: modello SVM addestrato.
        - X_val: dati di input di validazione.
        - y_val: etichette di validazione (stringhe).
        - X_test: dati di input di test.
        - y_test: etichette di test (stringhe).
    """
    # Codifica le etichette stringa in numeriche
    le = LabelEncoder()
    y_val_encoded = le.fit_transform(y_val)
    y_test_encoded = le.transform(y_test)

    # Predizioni del modello
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)

    # Codifica anche le predizioni
    y_val_pred_encoded = le.transform(y_val_pred)
    y_test_pred_encoded = le.transform(y_test_pred)

    # Genera le matrici di confusione
    cm_val = confusion_matrix(y_val_encoded, y_val_pred_encoded, labels=np.unique(y_val_encoded))
    cm_test = confusion_matrix(y_test_encoded, y_test_pred_encoded, labels=np.unique(y_test_encoded))

    # Genera le rappresentazioni grafiche
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Visualizzazione matrice di validazione
    disp_val = ConfusionMatrixDisplay(confusion_matrix=cm_val, display_labels=le.classes_)
    disp_val.plot(ax=axes[0], cmap='Greens', colorbar=False)
    axes[0].set_title("SVM - Matrice di Confusione (Validazione)")

    # Visualizzazione matrice di test
    disp_test = ConfusionMatrixDisplay(confusion_matrix=cm_test, display_labels=le.classes_)
    disp_test.plot(ax=axes[1], cmap='Greens', colorbar=False)
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
    disp_val.plot(cmap='Greens', ax=axes[0])
    axes[0].set_title("LightGBM - Matrice di Confusione (Validazione)")

    disp_test.plot(cmap='Greens', ax=axes[1])
    axes[1].set_title("LightGBM - Matrice di Confusione (Test)")

    # Rimuove la griglia
    for ax in axes:
        ax.grid(False)

    # Mostra le figure
    plt.tight_layout()
    plt.show()
