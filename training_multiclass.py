import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from imblearn.pipeline import Pipeline as imPipeline
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
import numpy as np
import joblib  # Per salvare i modelli

# Carica il CSV
df = pd.read_csv('audio_features.csv')

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

# Imputazione dei dati
X_train_imputed = imputer.fit_transform(X_train)
X_val_imputed = imputer.transform(X_val)
X_test_imputed = imputer.transform(X_test)

# Definisci SMOTE con un valore iniziale di k_neighbors
k_neighbors = 5  # Valore predefinito, da adattare se necessario
smote = SMOTE(sampling_strategy='auto', k_neighbors=k_neighbors, random_state=42)

# Gestione delle classi con pochi campioni
class_counts = pd.Series(y_train_encoded).value_counts()
min_samples_per_class = k_neighbors  # Numero minimo di campioni per usare SMOTE

# Escludi classi con meno di min_samples_per_class campioni
classes_to_keep = class_counts[class_counts >= min_samples_per_class].index

# Converti a DataFrame per filtraggio
X_train_df = pd.DataFrame(X_train_imputed)
y_train_df = pd.Series(y_train_encoded)

# Filtra i dati
X_train_filtered = X_train_df[y_train_df.isin(classes_to_keep)].values
y_train_filtered = y_train_df[y_train_df.isin(classes_to_keep)].values

# Crea l'istanza di SMOTE con k_neighbors adeguato
smote = SMOTE(sampling_strategy='auto', k_neighbors=min_samples_per_class, random_state=42)

# Applica SMOTE solo al set di addestramento
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_filtered, y_train_filtered)

# Trova tutte le etichette uniche nei set di dati
all_classes = np.unique(np.concatenate([y_train_encoded, y_val_encoded, y_test_encoded]))
class_labels = subclass_encoder.classes_

# Definisci i modelli da addestrare
models = {
    'Logistic Regression': imPipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(random_state=42, multi_class='ovr'))
    ]),
    'Random Forest': imPipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(random_state=42))
    ]),
    'SVM': imPipeline([
        ('scaler', StandardScaler()),
        ('classifier', SVC(random_state=42, decision_function_shape='ovr'))
    ]),
    'KNN': imPipeline([
        ('scaler', StandardScaler()),
        ('classifier', KNeighborsClassifier())
    ]),
    'Gradient Boosting': imPipeline([
        ('scaler', StandardScaler()),
        ('classifier', GradientBoostingClassifier(random_state=42))
    ])
}

# Addestra, valuta e salva i risultati dei modelli
for model_name, model_pipeline in models.items():
    # Addestra il modello
    model_pipeline.fit(X_train_resampled, y_train_resampled)

    # Valuta il modello sul set di validation
    y_val_pred = model_pipeline.predict(X_val_imputed)
    val_report = classification_report(y_val_encoded, y_val_pred, target_names=class_labels, labels=all_classes,
                                       zero_division=0, output_dict=True)

    # Valuta il modello sul set di testing
    y_test_pred = model_pipeline.predict(X_test_imputed)
    test_report = classification_report(y_test_encoded, y_test_pred, target_names=class_labels, labels=all_classes,
                                        zero_division=0, output_dict=True)

    # Salva i report in CSV
    val_report_df = pd.DataFrame(val_report).transpose()
    test_report_df = pd.DataFrame(test_report).transpose()

    val_report_df.to_csv(f'{model_name}_validation_report.csv', index=True)
    test_report_df.to_csv(f'{model_name}_test_report.csv', index=True)

    # Salva il modello
    joblib.dump(model_pipeline, f'{model_name}_model.pkl')

    print(f"\nModello: {model_name}")
    print("Report di validazione salvato in:", f'{model_name}_validation_report.csv')
    print("Report di test salvato in:", f'{model_name}_test_report.csv')
    print("Modello salvato in:", f'{model_name}_model.pkl')
