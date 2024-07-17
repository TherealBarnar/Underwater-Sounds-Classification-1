import pandas as pd

# Carica il dataset dal file CSV
dataset_path = 'audio_features.csv'  # Sostituisci con il percorso al tuo dataset
df = pd.read_csv(dataset_path)

# Controlla se il dataset è stato caricato correttamente
print("Numero di campioni nel dataset originale:", len(df))

# Mostra i nomi delle colonne del dataset
print("Nomi delle colonne nel dataset:", df.columns)

# Visualizza le prime righe del dataset per ispezionare il contenuto
print("Prime righe del dataset:")
print(df.head())

# Verifica se ci sono valori NaN nella colonna delle etichette
# Sostituisci 'label' con il nome corretto della colonna delle etichette nel tuo dataset
label_column = 'label'  # Assumi che la colonna si chiami 'label'
if label_column not in df.columns:
    raise ValueError(f"La colonna delle etichette '{label_column}' non esiste nel dataset.")

# Conta i valori NaN nella colonna delle etichette
num_nan_labels = df[label_column].isna().sum()
print(f"Numero di valori NaN nella colonna delle etichette '{label_column}':", num_nan_labels)

# Se ci sono troppi valori NaN, mostra un esempio di righe con NaN per ispezione
if num_nan_labels > 0:
    print("Esempio di righe con valori NaN nella colonna delle etichette:")
    print(df[df[label_column].isna()].head())

# Rimuovi le righe dove le etichette (y) sono NaN
df = df.dropna(subset=[label_column])

# Controlla il numero di campioni dopo la rimozione dei NaN
print("Numero di campioni dopo la rimozione dei NaN:", len(df))

# Se il dataset è vuoto dopo la rimozione, solleva un errore
if len(df) == 0:
    raise ValueError("Il dataset è vuoto dopo la rimozione dei NaN nelle etichette.")

# Separazione delle caratteristiche (X) e dell'etichetta (y)
X = df.drop(columns=label_column)  # Tutte le colonne tranne 'label' sono caratteristiche
y = df[label_column]  # La colonna 'label' è l'etichetta

# Verifica se ci sono ancora NaN nelle caratteristiche e nelle etichette
print("Valori NaN nelle caratteristiche (X):", X.isna().sum().sum())
print("Valori NaN nelle etichette (y):", y.isna().sum().sum())

# Controlla il numero di campioni in X e y
print("Numero di campioni in X:", len(X))
print("Numero di campioni in y:", len(y))

# Split del dataset in 80% training, 10% validation e 10% test
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# Applica SMOTE per il bilanciamento del training set
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

# Stampa delle dimensioni dei dataset per conferma
print("Dimensioni originali del training set:", X_train.shape, y_train.shape)
print("Dimensioni del training set bilanciato:", X_train_balanced.shape, y_train_balanced.shape)
print("Dimensioni del validation set:", X_val.shape, y_val.shape)
print("Dimensioni del test set:", X_test.shape, y_test.shape)
