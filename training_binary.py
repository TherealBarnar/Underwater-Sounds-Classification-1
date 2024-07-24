import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import joblib  # Importa joblib per salvare i modelli

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
train_df_filtered = train_df[train_df['Class'].isin(class_encoder.inverse_transform(classes_with_sufficient_samples))]
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

# Crea l'istanza di SMOTE
smote = SMOTE(random_state=42)

# Applica SMOTE solo al set di addestramento filtrato
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_imputed, y_train_encoded_filtered)

# Verifica la distribuzione delle classi dopo SMOTE
print(f"Distribuzione delle classi nel set di training dopo SMOTE:\n{pd.Series(y_train_resampled).value_counts()}")

# Definisci i modelli da addestrare
models = {
    'Logistic Regression': Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(random_state=42))
    ]),
    'Random Forest': Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(random_state=42))
    ]),
    'SVM': Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', SVC(random_state=42))
    ]),
    'KNN': Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', KNeighborsClassifier())
    ]),
    'Gradient Boosting': Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', GradientBoostingClassifier(random_state=42))
    ]),
    'XGBoost': Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', XGBClassifier(random_state=42))
    ])
}


# Funzione per salvare i report in file CSV
def save_classification_report(model_name, y_val_true, y_val_pred, y_test_true, y_test_pred):
    val_report = classification_report(y_val_true, y_val_pred, output_dict=True)
    test_report = classification_report(y_test_true, y_test_pred, output_dict=True)

    val_df = pd.DataFrame(val_report).transpose()
    test_df = pd.DataFrame(test_report).transpose()

    val_df.to_csv(f'{model_name.replace(" ", "_")}_validation_report.csv')
    test_df.to_csv(f'{model_name.replace(" ", "_")}_testing_report.csv')


# Addestra, valuta e salva i modelli
for model_name, model_pipeline in models.items():
    # Addestra il modello
    model_pipeline.fit(X_train_resampled, y_train_resampled)

    # Valuta il modello sul set di validation
    y_val_pred = model_pipeline.predict(X_val_imputed)

    # Valuta il modello sul set di testing
    y_test_pred = model_pipeline.predict(X_test_imputed)

    # Stampa i risultati
    print(f"\nAddestramento e valutazione del modello: {model_name}")

    print("Valutazione sul set di validation:")
    val_report = classification_report(y_val_encoded, y_val_pred, output_dict=True)
    print(f"Accuracy: {val_report['accuracy']:.2f}")
    print(
        f"Class 0 - Precision: {val_report['0']['precision']:.2f}, Recall: {val_report['0']['recall']:.2f}, F1-Score: {val_report['0']['f1-score']:.2f}")
    print(
        f"Class 1 - Precision: {val_report['1']['precision']:.2f}, Recall: {val_report['1']['recall']:.2f}, F1-Score: {val_report['1']['f1-score']:.2f}")

    print("Valutazione sul set di testing:")
    test_report = classification_report(y_test_encoded, y_test_pred, output_dict=True)
    print(f"Accuracy: {test_report['accuracy']:.2f}")
    print(
        f"Class 0 - Precision: {test_report['0']['precision']:.2f}, Recall: {test_report['0']['recall']:.2f}, F1-Score: {test_report['0']['f1-score']:.2f}")
    print(
        f"Class 1 - Precision: {test_report['1']['precision']:.2f}, Recall: {test_report['1']['recall']:.2f}, F1-Score: {test_report['1']['f1-score']:.2f}")

    # Salva i report in file CSV
    save_classification_report(model_name, y_val_encoded, y_val_pred, y_test_encoded, y_test_pred)

    # Salva il modello
    joblib.dump(model_pipeline, f'{model_name.replace(" ", "_")}_model.pkl')
    print(f"Modello '{model_name}' salvato come '{model_name.replace(' ', '_')}_model.pkl'")
