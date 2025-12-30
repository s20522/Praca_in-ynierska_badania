"""
Nowe eksperymenty - Inżynieria cech (Feature Engineering)
==========================================================

Autor: Heart Failure Research Team
Data: 2025-12-29

Cel: Przetestowanie różnych technik inżynierii cech i porównanie z modelem bazowym
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, classification_report)
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("NOWE EKSPERYMENTY - INŻYNIERIA CECH")
print("="*80)

# Wczytanie danych
df = pd.read_csv('../data/heart_failure_data.csv')
print(f"\nRozmiar zbioru danych: {df.shape[0]} wierszy, {df.shape[1]} kolumn")

# Mapowanie nazw kolumn
column_mapping = {
    'Age': 'age',
    'Ejection.Fraction': 'ejection_fraction',
    'Creatinine': 'serum_creatinine',
    'Sodium': 'serum_sodium',
    'CPK': 'creatinine_phosphokinase',
    'Pletelets': 'platelets',
    'BP': 'high_blood_pressure',
    'Anaemia': 'anaemia',
    'Diabetes': 'diabetes',
    'Gender': 'sex',
    'Smoking': 'smoking',
    'TIME': 'time',
    'Event': 'DEATH_EVENT'
}
df = df.rename(columns=column_mapping)
print(f"Kolumny po mapowaniu: {list(df.columns)}")

# Parametry stałe (z modelu bazowego)
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# Najlepsze parametry z modelu bazowego
BEST_PARAMS = {
    'n_estimators': 100,
    'max_depth': 50,
    'min_samples_split': 5,
    'min_samples_leaf': 8,
    'max_features': None,
    'class_weight': 'balanced',
    'bootstrap': True,
    'random_state': RANDOM_STATE
}

# Słownik do przechowywania wyników
results = {}

# ============================================================================
# EKSPERYMENT 0: MODEL BAZOWY (BASELINE)
# ============================================================================

print("\n" + "="*80)
print("EKSPERYMENT 0: MODEL BAZOWY (BASELINE)")
print("="*80)
print("Cechy: age, ejection_fraction, serum_creatinine (surowe)")
print("Normalizacja: StandardScaler")

# Przygotowanie danych bazowych
X_baseline = df[['age', 'ejection_fraction', 'serum_creatinine']].copy()
y = df['DEATH_EVENT'].copy()
print(f"\nKształt X_baseline: {X_baseline.shape}")
print(f"Kształt y: {y.shape}")

# Podział
X_train_base, X_test_base, y_train, y_test = train_test_split(
    X_baseline, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

# Standaryzacja
scaler_base = StandardScaler()
X_train_base_scaled = scaler_base.fit_transform(X_train_base)
X_test_base_scaled = scaler_base.transform(X_test_base)

# Trening
rf_baseline = RandomForestClassifier(**BEST_PARAMS)
rf_baseline.fit(X_train_base_scaled, y_train)

# Predykcja
y_pred_base = rf_baseline.predict(X_test_base_scaled)
y_pred_proba_base = rf_baseline.predict_proba(X_test_base_scaled)[:, 1]

# Metryki
results['Baseline'] = {
    'accuracy': accuracy_score(y_test, y_pred_base),
    'precision': precision_score(y_test, y_pred_base),
    'recall': recall_score(y_test, y_pred_base),
    'f1': f1_score(y_test, y_pred_base),
    'auc': roc_auc_score(y_test, y_pred_proba_base),
    'features': ['age', 'ejection_fraction', 'serum_creatinine'],
    'n_features': 3
}

print(f"\nWyniki:")
print(f"  Accuracy:  {results['Baseline']['accuracy']:.4f}")
print(f"  Precision: {results['Baseline']['precision']:.4f}")
print(f"  Recall:    {results['Baseline']['recall']:.4f}")
print(f"  F1-score:  {results['Baseline']['f1']:.4f}")
print(f"  AUC-ROC:   {results['Baseline']['auc']:.4f}")

# ============================================================================
# EKSPERYMENT 1: DYSKRETYZACJA CECH
# ============================================================================

print("\n" + "="*80)
print("EKSPERYMENT 1: DYSKRETYZACJA CECH")
print("="*80)
print("Podział cech ciągłych na kategorie kliniczne")

# Przygotowanie danych z dyskretyzacją
X_discrete = df[['age', 'ejection_fraction', 'serum_creatinine']].copy()

# Dyskretyzacja Age (wiek)
# Przedziały: [40-60), [60-80), [80-95]
X_discrete['age_cat'] = pd.cut(
    X_discrete['age'], 
    bins=[0, 60, 80, 100], 
    labels=[0, 1, 2]
).astype(int)

# Dyskretyzacja Ejection Fraction (frakcja wyrzutowa)
# Przedziały: ciężka (<30%), umiarkowana (30-45%), lekka/norma (>45%)
X_discrete['ef_cat'] = pd.cut(
    X_discrete['ejection_fraction'], 
    bins=[0, 30, 45, 100], 
    labels=[0, 1, 2]
).astype(int)

# Dyskretyzacja Serum Creatinine (kreatynina)
# Przedziały: norma (0.5-1.2), podwyższony (1.2-3.0), wysoki (>3.0)
X_discrete['creat_cat'] = pd.cut(
    X_discrete['serum_creatinine'], 
    bins=[0, 1.2, 3.0, 10], 
    labels=[0, 1, 2]
).astype(int)

# Użycie tylko kategorii
X_discrete_only = X_discrete[['age_cat', 'ef_cat', 'creat_cat']]

print(f"\nRozkład kategorii:")
print(f"Age categories: {X_discrete_only['age_cat'].value_counts().sort_index().to_dict()}")
print(f"EF categories: {X_discrete_only['ef_cat'].value_counts().sort_index().to_dict()}")
print(f"Creatinine categories: {X_discrete_only['creat_cat'].value_counts().sort_index().to_dict()}")

# Podział
X_train_disc, X_test_disc, _, _ = train_test_split(
    X_discrete_only, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

# Standaryzacja (nawet dla kategorii, dla spójności)
scaler_disc = StandardScaler()
X_train_disc_scaled = scaler_disc.fit_transform(X_train_disc)
X_test_disc_scaled = scaler_disc.transform(X_test_disc)

# Trening
rf_discrete = RandomForestClassifier(**BEST_PARAMS)
rf_discrete.fit(X_train_disc_scaled, y_train)

# Predykcja
y_pred_disc = rf_discrete.predict(X_test_disc_scaled)
y_pred_proba_disc = rf_discrete.predict_proba(X_test_disc_scaled)[:, 1]

# Metryki
results['Discretization'] = {
    'accuracy': accuracy_score(y_test, y_pred_disc),
    'precision': precision_score(y_test, y_pred_disc),
    'recall': recall_score(y_test, y_pred_disc),
    'f1': f1_score(y_test, y_pred_disc),
    'auc': roc_auc_score(y_test, y_pred_proba_disc),
    'features': ['age_cat', 'ef_cat', 'creat_cat'],
    'n_features': 3
}

print(f"\nWyniki:")
print(f"  Accuracy:  {results['Discretization']['accuracy']:.4f}")
print(f"  Precision: {results['Discretization']['precision']:.4f}")
print(f"  Recall:    {results['Discretization']['recall']:.4f}")
print(f"  F1-score:  {results['Discretization']['f1']:.4f}")
print(f"  AUC-ROC:   {results['Discretization']['auc']:.4f}")

# ============================================================================
# EKSPERYMENT 2: CECHY INTERAKCYJNE
# ============================================================================

print("\n" + "="*80)
print("EKSPERYMENT 2: CECHY INTERAKCYJNE")
print("="*80)
print("Dodanie cech reprezentujących interakcje między zmiennymi")

# Przygotowanie danych z interakcjami
X_interact = df[['age', 'ejection_fraction', 'serum_creatinine']].copy()

# Tworzenie cech interakcyjnych
X_interact['age_x_creat'] = X_interact['age'] * X_interact['serum_creatinine']
X_interact['ef_x_sodium'] = df['ejection_fraction'] * df['serum_sodium']
X_interact['age_x_ef'] = X_interact['age'] * X_interact['ejection_fraction']

print(f"\nCechy: {list(X_interact.columns)}")
print(f"Liczba cech: {X_interact.shape[1]}")

# Podział
X_train_int, X_test_int, _, _ = train_test_split(
    X_interact, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

# Standaryzacja
scaler_int = StandardScaler()
X_train_int_scaled = scaler_int.fit_transform(X_train_int)
X_test_int_scaled = scaler_int.transform(X_test_int)

# Trening
rf_interact = RandomForestClassifier(**BEST_PARAMS)
rf_interact.fit(X_train_int_scaled, y_train)

# Predykcja
y_pred_int = rf_interact.predict(X_test_int_scaled)
y_pred_proba_int = rf_interact.predict_proba(X_test_int_scaled)[:, 1]

# Metryki
results['Interactions'] = {
    'accuracy': accuracy_score(y_test, y_pred_int),
    'precision': precision_score(y_test, y_pred_int),
    'recall': recall_score(y_test, y_pred_int),
    'f1': f1_score(y_test, y_pred_int),
    'auc': roc_auc_score(y_test, y_pred_proba_int),
    'features': list(X_interact.columns),
    'n_features': X_interact.shape[1]
}

print(f"\nWyniki:")
print(f"  Accuracy:  {results['Interactions']['accuracy']:.4f}")
print(f"  Precision: {results['Interactions']['precision']:.4f}")
print(f"  Recall:    {results['Interactions']['recall']:.4f}")
print(f"  F1-score:  {results['Interactions']['f1']:.4f}")
print(f"  AUC-ROC:   {results['Interactions']['auc']:.4f}")

# Feature importance dla interakcji
feature_imp_int = rf_interact.feature_importances_
print(f"\nWażność cech (z interakcjami):")
for feat, imp in zip(X_interact.columns, feature_imp_int):
    print(f"  {feat}: {imp:.4f}")

# ============================================================================
# EKSPERYMENT 3: MINMAX NORMALIZACJA
# ============================================================================

print("\n" + "="*80)
print("EKSPERYMENT 3: MINMAX NORMALIZACJA")
print("="*80)
print("Użycie MinMaxScaler zamiast StandardScaler")

# Użycie tych samych cech bazowych
X_train_mm = X_train_base.copy()
X_test_mm = X_test_base.copy()

# MinMax normalizacja
scaler_mm = MinMaxScaler()
X_train_mm_scaled = scaler_mm.fit_transform(X_train_mm)
X_test_mm_scaled = scaler_mm.transform(X_test_mm)

# Trening
rf_minmax = RandomForestClassifier(**BEST_PARAMS)
rf_minmax.fit(X_train_mm_scaled, y_train)

# Predykcja
y_pred_mm = rf_minmax.predict(X_test_mm_scaled)
y_pred_proba_mm = rf_minmax.predict_proba(X_test_mm_scaled)[:, 1]

# Metryki
results['MinMax'] = {
    'accuracy': accuracy_score(y_test, y_pred_mm),
    'precision': precision_score(y_test, y_pred_mm),
    'recall': recall_score(y_test, y_pred_mm),
    'f1': f1_score(y_test, y_pred_mm),
    'auc': roc_auc_score(y_test, y_pred_proba_mm),
    'features': ['age', 'ejection_fraction', 'serum_creatinine'],
    'n_features': 3
}

print(f"\nWyniki:")
print(f"  Accuracy:  {results['MinMax']['accuracy']:.4f}")
print(f"  Precision: {results['MinMax']['precision']:.4f}")
print(f"  Recall:    {results['MinMax']['recall']:.4f}")
print(f"  F1-score:  {results['MinMax']['f1']:.4f}")
print(f"  AUC-ROC:   {results['MinMax']['auc']:.4f}")

# ============================================================================
# EKSPERYMENT 4: WSZYSTKIE CECHY + INTERAKCJE
# ============================================================================

print("\n" + "="*80)
print("EKSPERYMENT 4: WSZYSTKIE CECHY + INTERAKCJE")
print("="*80)
print("Połączenie wszystkich cech oryginalnych z interakcjami")

# Przygotowanie danych ze wszystkimi cechami (bez 'time')
all_features = ['age', 'anaemia', 'creatinine_phosphokinase', 'diabetes',
                'ejection_fraction', 'high_blood_pressure', 'platelets',
                'serum_creatinine', 'serum_sodium', 'sex', 'smoking']

X_all = df[all_features].copy()

# Dodanie interakcji
X_all['age_x_creat'] = X_all['age'] * X_all['serum_creatinine']
X_all['ef_x_sodium'] = X_all['ejection_fraction'] * X_all['serum_sodium']
X_all['age_x_ef'] = X_all['age'] * X_all['ejection_fraction']

print(f"\nLiczba cech: {X_all.shape[1]}")

# Podział
X_train_all, X_test_all, _, _ = train_test_split(
    X_all, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

# Standaryzacja
scaler_all = StandardScaler()
X_train_all_scaled = scaler_all.fit_transform(X_train_all)
X_test_all_scaled = scaler_all.transform(X_test_all)

# Trening
rf_all = RandomForestClassifier(**BEST_PARAMS)
rf_all.fit(X_train_all_scaled, y_train)

# Predykcja
y_pred_all = rf_all.predict(X_test_all_scaled)
y_pred_proba_all = rf_all.predict_proba(X_test_all_scaled)[:, 1]

# Metryki
results['All_Features'] = {
    'accuracy': accuracy_score(y_test, y_pred_all),
    'precision': precision_score(y_test, y_pred_all),
    'recall': recall_score(y_test, y_pred_all),
    'f1': f1_score(y_test, y_pred_all),
    'auc': roc_auc_score(y_test, y_pred_proba_all),
    'features': list(X_all.columns),
    'n_features': X_all.shape[1]
}

print(f"\nWyniki:")
print(f"  Accuracy:  {results['All_Features']['accuracy']:.4f}")
print(f"  Precision: {results['All_Features']['precision']:.4f}")
print(f"  Recall:    {results['All_Features']['recall']:.4f}")
print(f"  F1-score:  {results['All_Features']['f1']:.4f}")
print(f"  AUC-ROC:   {results['All_Features']['auc']:.4f}")

# Top 10 cech
feature_imp_all = rf_all.feature_importances_
top_10_idx = np.argsort(feature_imp_all)[-10:][::-1]
print(f"\nTop 10 najważniejszych cech:")
for idx in top_10_idx:
    print(f"  {X_all.columns[idx]}: {feature_imp_all[idx]:.4f}")

# ============================================================================
# PODSUMOWANIE WYNIKÓW
# ============================================================================

print("\n" + "="*80)
print("PODSUMOWANIE WSZYSTKICH EKSPERYMENTÓW")
print("="*80)

# Tabela porównawcza
comparison_df = pd.DataFrame(results).T
comparison_df = comparison_df[['accuracy', 'precision', 'recall', 'f1', 'auc', 'n_features']]
comparison_df = comparison_df.round(4)

print("\n" + comparison_df.to_string())

# Zapisanie wyników
comparison_df.to_csv('../results/feature_engineering_comparison.csv')
print("\n✓ Zapisano: results/feature_engineering_comparison.csv")

# Zapisanie szczegółowych wyników
import json
with open('../results/feature_engineering_details.json', 'w') as f:
    json.dump(results, f, indent=2)
print("✓ Zapisano: results/feature_engineering_details.json")

# Zapisanie predykcji dla najlepszego modelu
best_model_name = comparison_df['f1'].idxmax()
print(f"\n🏆 Najlepszy model (F1-score): {best_model_name}")
print(f"   F1-score: {comparison_df.loc[best_model_name, 'f1']:.4f}")
print(f"   Recall: {comparison_df.loc[best_model_name, 'recall']:.4f}")
print(f"   AUC-ROC: {comparison_df.loc[best_model_name, 'auc']:.4f}")

# Zapisanie danych do wizualizacji
np.save('../results/fe_y_test.npy', y_test)

# Zapisanie predykcji dla każdego modelu
np.save('../results/fe_baseline_pred.npy', y_pred_base)
np.save('../results/fe_baseline_proba.npy', y_pred_proba_base)
np.save('../results/fe_discrete_pred.npy', y_pred_disc)
np.save('../results/fe_discrete_proba.npy', y_pred_proba_disc)
np.save('../results/fe_interact_pred.npy', y_pred_int)
np.save('../results/fe_interact_proba.npy', y_pred_proba_int)
np.save('../results/fe_minmax_pred.npy', y_pred_mm)
np.save('../results/fe_minmax_proba.npy', y_pred_proba_mm)
np.save('../results/fe_all_pred.npy', y_pred_all)
np.save('../results/fe_all_proba.npy', y_pred_proba_all)

print("\n" + "="*80)
print("EKSPERYMENTY ZAKOŃCZONE POMYŚLNIE")
print("="*80)
