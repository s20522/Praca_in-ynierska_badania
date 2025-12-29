"""
Reprodukcja modelu Random Forest - Heart Failure Prediction
============================================================

Autor: Heart Failure Research Team
Data: 2025-12-29

Cel: Reprodukcja modelu Random Forest z publikacji bazowej, zgodnie z metodyką:
- Wybór cech: age, ejection_fraction, serum_creatinine (bez 'time' - target leakage)
- Randomized Cross-Validation do optymalizacji hiperparametrów
- Ewaluacja z metrykami: Accuracy, Precision, Recall, F1-score, AUC-ROC
- Porównanie z wynikami z publikacji
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                             roc_auc_score, confusion_matrix, classification_report,
                             roc_curve, precision_recall_curve, auc)
import warnings
warnings.filterwarnings('ignore')

# Konfiguracja matplotlib
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'

print("="*80)
print("REPRODUKCJA MODELU RANDOM FOREST")
print("="*80)

# ============================================================================
# 1. WCZYTANIE I PRZYGOTOWANIE DANYCH
# ============================================================================

print("\n" + "="*80)
print("ETAP 1: WCZYTANIE I PRZYGOTOWANIE DANYCH")
print("="*80)

# Wczytanie danych
df = pd.read_csv('../data/heart_failure_data.csv')

# Mapowanie nazw kolumn
column_mapping = {
    'TIME': 'time',
    'Event': 'DEATH_EVENT',
    'Gender': 'sex',
    'Smoking': 'smoking',
    'Diabetes': 'diabetes',
    'BP': 'high_blood_pressure',
    'Anaemia': 'anaemia',
    'Age': 'age',
    'Ejection.Fraction': 'ejection_fraction',
    'Sodium': 'serum_sodium',
    'Creatinine': 'serum_creatinine',
    'Pletelets': 'platelets',
    'CPK': 'creatinine_phosphokinase'
}

df = df.rename(columns=column_mapping)

print(f"\nRozmiar zbioru danych: {df.shape[0]} wierszy, {df.shape[1]} kolumn")
print(f"Rozkład klasy celu: {df['DEATH_EVENT'].value_counts().to_dict()}")

# Wybór cech zgodnie z publikacją (najważniejsze cechy z analizy Coksa)
# WYKLUCZAMY 'time' ze względu na target leakage!
selected_features = ['age', 'ejection_fraction', 'serum_creatinine']

print(f"\nWybrane cechy do modelowania: {selected_features}")
print("Uzasadnienie: Najwyższa istotność statystyczna w analizie Coksa (bez 'time')")

X = df[selected_features]
y = df['DEATH_EVENT']

print(f"\nKształt macierzy cech (X): {X.shape}")
print(f"Kształt wektora celu (y): {y.shape}")

# ============================================================================
# 2. PODZIAŁ DANYCH
# ============================================================================

print("\n" + "="*80)
print("ETAP 2: PODZIAŁ DANYCH")
print("="*80)

# Podział na zbiór treningowy i testowy (80/20)
# stratify=y zapewnia zachowanie proporcji klas w obu zbiorach
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nZbiór treningowy: {X_train.shape[0]} próbek")
print(f"Zbiór testowy: {X_test.shape[0]} próbek")
print(f"\nRozkład klas w zbiorze treningowym:")
print(y_train.value_counts())
print(f"\nRozkład klas w zbiorze testowym:")
print(y_test.value_counts())

# ============================================================================
# 3. NORMALIZACJA DANYCH
# ============================================================================

print("\n" + "="*80)
print("ETAP 3: NORMALIZACJA DANYCH")
print("="*80)

# Standaryzacja cech (średnia=0, odchylenie std=1)
# Ważne: dopasowujemy scaler TYLKO na zbiorze treningowym!
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nZastosowano StandardScaler (standaryzacja)")
print("Średnie po standaryzacji (zbiór treningowy):")
print(np.mean(X_train_scaled, axis=0))
print("\nOdchylenia standardowe po standaryzacji (zbiór treningowy):")
print(np.std(X_train_scaled, axis=0))

# ============================================================================
# 4. OPTYMALIZACJA HIPERPARAMETRÓW - RANDOMIZED SEARCH CV
# ============================================================================

print("\n" + "="*80)
print("ETAP 4: OPTYMALIZACJA HIPERPARAMETRÓW")
print("="*80)

# Definicja przestrzeni hiperparametrów do przeszukania
param_distributions = {
    'n_estimators': [50, 100, 200, 300, 500],
    'max_depth': [None, 10, 20, 30, 40, 50],
    'min_samples_split': [2, 5, 10, 15],
    'min_samples_leaf': [1, 2, 4, 8],
    'max_features': ['sqrt', 'log2', None],
    'bootstrap': [True, False],
    'class_weight': ['balanced', 'balanced_subsample', None]
}

print("\nPrzestrzeń hiperparametrów:")
for param, values in param_distributions.items():
    print(f"  {param}: {values}")

# Random Forest z domyślnymi parametrami jako punkt startowy
rf_base = RandomForestClassifier(random_state=42)

# Stratified K-Fold (5 foldów) - zachowuje proporcje klas
cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Randomized Search CV
print("\nRozpoczynanie Randomized Search CV...")
print("Liczba iteracji: 100")
print("Strategia walidacji krzyżowej: Stratified 5-Fold")

random_search = RandomizedSearchCV(
    estimator=rf_base,
    param_distributions=param_distributions,
    n_iter=100,  # Liczba losowych kombinacji do przetestowania
    cv=cv_strategy,
    scoring='f1',  # Optymalizujemy F1-score (ważne przy niezbalansowaniu)
    n_jobs=-1,  # Użyj wszystkich dostępnych rdzeni
    random_state=42,
    verbose=1
)

random_search.fit(X_train_scaled, y_train)

print("\n✓ Optymalizacja zakończona!")
print(f"\nNajlepsze parametry:")
for param, value in random_search.best_params_.items():
    print(f"  {param}: {value}")
print(f"\nNajlepszy F1-score (CV): {random_search.best_score_:.4f}")

# Najlepszy model
best_rf = random_search.best_estimator_

# ============================================================================
# 5. TRENING FINALNEGO MODELU
# ============================================================================

print("\n" + "="*80)
print("ETAP 5: TRENING FINALNEGO MODELU")
print("="*80)

print("\nTrening modelu Random Forest z optymalnymi hiperparametrami...")
# Model już wytrenowany przez RandomizedSearchCV, ale możemy go ponownie wytrenować
best_rf.fit(X_train_scaled, y_train)
print("✓ Trening zakończony!")

# ============================================================================
# 6. PREDYKCJA I EWALUACJA
# ============================================================================

print("\n" + "="*80)
print("ETAP 6: PREDYKCJA I EWALUACJA")
print("="*80)

# Predykcje na zbiorze testowym
y_pred = best_rf.predict(X_test_scaled)
y_pred_proba = best_rf.predict_proba(X_test_scaled)[:, 1]

# Obliczenie metryk
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print("\n" + "-"*80)
print("WYNIKI NA ZBIORZE TESTOWYM")
print("-"*80)
print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"Precision: {precision:.4f} ({precision*100:.2f}%)")
print(f"Recall:    {recall:.4f} ({recall*100:.2f}%)")
print(f"F1-score:  {f1:.4f} ({f1*100:.2f}%)")
print(f"AUC-ROC:   {roc_auc:.4f}")

# Macierz pomyłek
cm = confusion_matrix(y_test, y_pred)
print("\n" + "-"*80)
print("MACIERZ POMYŁEK")
print("-"*80)
print(f"True Negatives (TN):  {cm[0,0]}")
print(f"False Positives (FP): {cm[0,1]}")
print(f"False Negatives (FN): {cm[1,0]}")
print(f"True Positives (TP):  {cm[1,1]}")

# Raport klasyfikacji
print("\n" + "-"*80)
print("RAPORT KLASYFIKACJI")
print("-"*80)
print(classification_report(y_test, y_pred, target_names=['Przeżył', 'Zmarł']))

# ============================================================================
# 7. WALIDACJA KRZYŻOWA NA CAŁYM ZBIORZE
# ============================================================================

print("\n" + "="*80)
print("ETAP 7: WALIDACJA KRZYŻOWA (5-FOLD)")
print("="*80)

# Przygotowanie pełnego zbioru (znormalizowanego)
X_full_scaled = scaler.fit_transform(X)

# 5-Fold Cross-Validation
cv_scores_accuracy = cross_val_score(best_rf, X_full_scaled, y, cv=5, scoring='accuracy')
cv_scores_f1 = cross_val_score(best_rf, X_full_scaled, y, cv=5, scoring='f1')
cv_scores_roc_auc = cross_val_score(best_rf, X_full_scaled, y, cv=5, scoring='roc_auc')

print("\nWyniki walidacji krzyżowej (5-Fold):")
print(f"\nAccuracy:")
print(f"  Średnia: {cv_scores_accuracy.mean():.4f} ± {cv_scores_accuracy.std():.4f}")
print(f"  Poszczególne foldy: {cv_scores_accuracy}")

print(f"\nF1-score:")
print(f"  Średnia: {cv_scores_f1.mean():.4f} ± {cv_scores_f1.std():.4f}")
print(f"  Poszczególne foldy: {cv_scores_f1}")

print(f"\nAUC-ROC:")
print(f"  Średnia: {cv_scores_roc_auc.mean():.4f} ± {cv_scores_roc_auc.std():.4f}")
print(f"  Poszczególne foldy: {cv_scores_roc_auc}")

# ============================================================================
# 8. WAŻNOŚĆ CECH (FEATURE IMPORTANCE)
# ============================================================================

print("\n" + "="*80)
print("ETAP 8: WAŻNOŚĆ CECH")
print("="*80)

feature_importances = best_rf.feature_importances_
feature_names = selected_features

print("\nWażność cech (Feature Importance):")
for name, importance in zip(feature_names, feature_importances):
    print(f"  {name}: {importance:.4f} ({importance*100:.2f}%)")

# ============================================================================
# 9. ZAPISANIE WYNIKÓW DO PLIKU
# ============================================================================

print("\n" + "="*80)
print("ETAP 9: ZAPISANIE WYNIKÓW")
print("="*80)

results = {
    'Model': 'Random Forest',
    'Features': selected_features,
    'Best_Params': random_search.best_params_,
    'Test_Accuracy': accuracy,
    'Test_Precision': precision,
    'Test_Recall': recall,
    'Test_F1': f1,
    'Test_AUC_ROC': roc_auc,
    'CV_Accuracy_Mean': cv_scores_accuracy.mean(),
    'CV_Accuracy_Std': cv_scores_accuracy.std(),
    'CV_F1_Mean': cv_scores_f1.mean(),
    'CV_F1_Std': cv_scores_f1.std(),
    'CV_AUC_ROC_Mean': cv_scores_roc_auc.mean(),
    'CV_AUC_ROC_Std': cv_scores_roc_auc.std(),
    'Feature_Importances': dict(zip(feature_names, feature_importances))
}

# Zapisanie do pliku tekstowego
with open('../results/random_forest_results.txt', 'w', encoding='utf-8') as f:
    f.write("="*80 + "\n")
    f.write("WYNIKI MODELU RANDOM FOREST\n")
    f.write("="*80 + "\n\n")
    
    f.write("WYBRANE CECHY:\n")
    f.write(f"{selected_features}\n\n")
    
    f.write("OPTYMALNE HIPERPARAMETRY:\n")
    for param, value in random_search.best_params_.items():
        f.write(f"  {param}: {value}\n")
    f.write("\n")
    
    f.write("WYNIKI NA ZBIORZE TESTOWYM:\n")
    f.write(f"  Accuracy:  {accuracy:.4f}\n")
    f.write(f"  Precision: {precision:.4f}\n")
    f.write(f"  Recall:    {recall:.4f}\n")
    f.write(f"  F1-score:  {f1:.4f}\n")
    f.write(f"  AUC-ROC:   {roc_auc:.4f}\n\n")
    
    f.write("WALIDACJA KRZYŻOWA (5-FOLD):\n")
    f.write(f"  Accuracy:  {cv_scores_accuracy.mean():.4f} ± {cv_scores_accuracy.std():.4f}\n")
    f.write(f"  F1-score:  {cv_scores_f1.mean():.4f} ± {cv_scores_f1.std():.4f}\n")
    f.write(f"  AUC-ROC:   {cv_scores_roc_auc.mean():.4f} ± {cv_scores_roc_auc.std():.4f}\n\n")
    
    f.write("WAŻNOŚĆ CECH:\n")
    for name, importance in zip(feature_names, feature_importances):
        f.write(f"  {name}: {importance:.4f}\n")

print("\n✓ Wyniki zapisane do: results/random_forest_results.txt")

# Zapisanie danych do wizualizacji
np.save('../results/rf_y_test.npy', y_test)
np.save('../results/rf_y_pred.npy', y_pred)
np.save('../results/rf_y_pred_proba.npy', y_pred_proba)
np.save('../results/rf_feature_importances.npy', feature_importances)

print("✓ Dane do wizualizacji zapisane")

print("\n" + "="*80)
print("REPRODUKCJA MODELU RANDOM FOREST ZAKOŃCZONA POMYŚLNIE")
print("="*80)
