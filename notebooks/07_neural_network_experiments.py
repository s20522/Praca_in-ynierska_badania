"""
Eksperymenty z Sieciami Neuronowymi (MLP)
==========================================

Autor: Heart Failure Research Team
Data: 2025-12-29

Cel: Przetestowanie różnych architektur, funkcji aktywacji, regularyzacji i optymalizatorów
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers, regularizers, callbacks
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, classification_report)
import warnings
warnings.filterwarnings('ignore')

# Ustawienie seed dla reprodukowalności
np.random.seed(42)
tf.random.set_seed(42)

print("="*80)
print("EKSPERYMENTY Z SIECIAMI NEURONOWYMI (MLP)")
print("="*80)
print(f"TensorFlow version: {tf.__version__}")

# Wczytanie danych
df = pd.read_csv('../data/heart_failure_data.csv')

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

# Parametry stałe
RANDOM_STATE = 42
TEST_SIZE = 0.2
VALIDATION_SPLIT = 0.2
EPOCHS = 100
BATCH_SIZE = 16

# Przygotowanie danych (3 kluczowe cechy, jak w Random Forest)
X = df[['age', 'ejection_fraction', 'serum_creatinine']].values
y = df['DEATH_EVENT'].values

# Podział na train/test
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

# Podział train na train/validation
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=VALIDATION_SPLIT, 
    random_state=RANDOM_STATE, stratify=y_train_full
)

print(f"\nRozmiary zbiorów:")
print(f"  Train: {X_train.shape[0]} próbek")
print(f"  Validation: {X_val.shape[0]} próbek")
print(f"  Test: {X_test.shape[0]} próbek")

# Standaryzacja
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Słownik do przechowywania wyników
results = {}

# Funkcja pomocnicza do budowy modelu
def build_model(architecture, activation='relu', dropout_rate=0.0, l2_reg=0.0):
    """
    Buduje model MLP z zadaną architekturą
    
    Args:
        architecture: lista z liczbą neuronów w każdej warstwie ukrytej
        activation: funkcja aktywacji ('relu', 'leaky_relu', 'elu')
        dropout_rate: współczynnik dropout
        l2_reg: współczynnik regularyzacji L2
    """
    model = keras.Sequential()
    model.add(layers.Input(shape=(3,)))
    
    for i, units in enumerate(architecture):
        if l2_reg > 0:
            model.add(layers.Dense(units, kernel_regularizer=regularizers.l2(l2_reg)))
        else:
            model.add(layers.Dense(units))
        
        # Funkcja aktywacji
        if activation == 'relu':
            model.add(layers.Activation('relu'))
        elif activation == 'leaky_relu':
            model.add(layers.LeakyReLU(alpha=0.01))
        elif activation == 'elu':
            model.add(layers.ELU(alpha=1.0))
        
        # Dropout
        if dropout_rate > 0:
            model.add(layers.Dropout(dropout_rate))
    
    # Warstwa wyjściowa
    model.add(layers.Dense(1, activation='sigmoid'))
    
    return model

# Funkcja pomocnicza do treningu i ewaluacji
def train_and_evaluate(model, optimizer, model_name, verbose=0):
    """Trenuje model i zwraca metryki"""
    
    # Kompilacja
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.AUC(name='auc')]
    )
    
    # Callbacks
    early_stop = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=0
    )
    
    # Trening
    history = model.fit(
        X_train_scaled, y_train,
        validation_data=(X_val_scaled, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stop],
        verbose=verbose
    )
    
    # Predykcja
    y_pred_proba = model.predict(X_test_scaled, verbose=0).flatten()
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    # Metryki
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'auc': roc_auc_score(y_test, y_pred_proba),
        'epochs_trained': len(history.history['loss']),
        'final_train_loss': history.history['loss'][-1],
        'final_val_loss': history.history['val_loss'][-1]
    }
    
    return metrics, history, y_pred, y_pred_proba

# ============================================================================
# EKSPERYMENT 1: RÓŻNE ARCHITEKTURY
# ============================================================================

print("\n" + "="*80)
print("EKSPERYMENT 1: RÓŻNE ARCHITEKTURY SIECI")
print("="*80)

architectures = {
    'Shallow_32': [32],
    'Shallow_64': [64],
    'Shallow_128': [128],
    'Medium_64_32': [64, 32],
    'Medium_128_64': [128, 64],
    'Deep_128_64_32': [128, 64, 32],
    'Deep_256_128_64': [256, 128, 64]
}

for name, arch in architectures.items():
    print(f"\n[{name}] Architektura: {arch}")
    model = build_model(arch, activation='relu', dropout_rate=0.0, l2_reg=0.0)
    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    
    metrics, history, y_pred, y_pred_proba = train_and_evaluate(
        model, optimizer, name, verbose=0
    )
    
    results[f'Arch_{name}'] = metrics
    
    print(f"  F1-score: {metrics['f1']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    print(f"  AUC-ROC: {metrics['auc']:.4f}")
    print(f"  Epochs: {metrics['epochs_trained']}")
    
    # Zapisanie predykcji dla najlepszego modelu
    if name == 'Medium_128_64':  # Zapisujemy dla typowej architektury
        np.save('../results/nn_arch_pred.npy', y_pred)
        np.save('../results/nn_arch_proba.npy', y_pred_proba)

# ============================================================================
# EKSPERYMENT 2: FUNKCJE AKTYWACJI
# ============================================================================

print("\n" + "="*80)
print("EKSPERYMENT 2: FUNKCJE AKTYWACJI")
print("="*80)

activations = ['relu', 'leaky_relu', 'elu']
base_arch = [128, 64]

for act in activations:
    print(f"\n[{act.upper()}]")
    model = build_model(base_arch, activation=act, dropout_rate=0.0, l2_reg=0.0)
    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    
    metrics, history, y_pred, y_pred_proba = train_and_evaluate(
        model, optimizer, f'Activation_{act}', verbose=0
    )
    
    results[f'Activation_{act}'] = metrics
    
    print(f"  F1-score: {metrics['f1']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    print(f"  AUC-ROC: {metrics['auc']:.4f}")
    
    # Zapisanie predykcji dla ReLU
    if act == 'relu':
        np.save('../results/nn_act_pred.npy', y_pred)
        np.save('../results/nn_act_proba.npy', y_pred_proba)

# ============================================================================
# EKSPERYMENT 3: REGULARYZACJA (DROPOUT)
# ============================================================================

print("\n" + "="*80)
print("EKSPERYMENT 3: REGULARYZACJA - DROPOUT")
print("="*80)

dropout_rates = [0.0, 0.2, 0.3, 0.5]

for dropout in dropout_rates:
    print(f"\n[Dropout={dropout}]")
    model = build_model(base_arch, activation='relu', dropout_rate=dropout, l2_reg=0.0)
    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    
    metrics, history, y_pred, y_pred_proba = train_and_evaluate(
        model, optimizer, f'Dropout_{dropout}', verbose=0
    )
    
    results[f'Dropout_{dropout}'] = metrics
    
    print(f"  F1-score: {metrics['f1']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    print(f"  AUC-ROC: {metrics['auc']:.4f}")
    
    # Zapisanie predykcji dla dropout=0.3
    if dropout == 0.3:
        np.save('../results/nn_dropout_pred.npy', y_pred)
        np.save('../results/nn_dropout_proba.npy', y_pred_proba)

# ============================================================================
# EKSPERYMENT 4: REGULARYZACJA L2
# ============================================================================

print("\n" + "="*80)
print("EKSPERYMENT 4: REGULARYZACJA L2")
print("="*80)

l2_values = [0.0, 0.001, 0.01, 0.1]

for l2 in l2_values:
    print(f"\n[L2={l2}]")
    model = build_model(base_arch, activation='relu', dropout_rate=0.0, l2_reg=l2)
    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    
    metrics, history, y_pred, y_pred_proba = train_and_evaluate(
        model, optimizer, f'L2_{l2}', verbose=0
    )
    
    results[f'L2_{l2}'] = metrics
    
    print(f"  F1-score: {metrics['f1']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    print(f"  AUC-ROC: {metrics['auc']:.4f}")
    
    # Zapisanie predykcji dla l2=0.01
    if l2 == 0.01:
        np.save('../results/nn_l2_pred.npy', y_pred)
        np.save('../results/nn_l2_proba.npy', y_pred_proba)

# ============================================================================
# EKSPERYMENT 5: OPTYMALIZATORY
# ============================================================================

print("\n" + "="*80)
print("EKSPERYMENT 5: OPTYMALIZATORY")
print("="*80)

optimizers_dict = {
    'Adam': keras.optimizers.Adam(learning_rate=0.001),
    'SGD': keras.optimizers.SGD(learning_rate=0.01, momentum=0.9),
    'RMSprop': keras.optimizers.RMSprop(learning_rate=0.001)
}

for opt_name, optimizer in optimizers_dict.items():
    print(f"\n[{opt_name}]")
    model = build_model(base_arch, activation='relu', dropout_rate=0.3, l2_reg=0.01)
    
    metrics, history, y_pred, y_pred_proba = train_and_evaluate(
        model, optimizer, f'Optimizer_{opt_name}', verbose=0
    )
    
    results[f'Optimizer_{opt_name}'] = metrics
    
    print(f"  F1-score: {metrics['f1']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    print(f"  AUC-ROC: {metrics['auc']:.4f}")
    
    # Zapisanie predykcji dla Adam
    if opt_name == 'Adam':
        np.save('../results/nn_opt_pred.npy', y_pred)
        np.save('../results/nn_opt_proba.npy', y_pred_proba)

# ============================================================================
# NAJLEPSZY MODEL - POŁĄCZENIE NAJLEPSZYCH USTAWIEŃ
# ============================================================================

print("\n" + "="*80)
print("EKSPERYMENT 6: NAJLEPSZY MODEL (OPTYMALNA KONFIGURACJA)")
print("="*80)

print("\nTrening najlepszego modelu z optymalnymi hiperparametrami...")
best_model = build_model(
    architecture=[128, 64],
    activation='relu',
    dropout_rate=0.3,
    l2_reg=0.01
)
optimizer = keras.optimizers.Adam(learning_rate=0.001)

metrics, history, y_pred, y_pred_proba = train_and_evaluate(
    best_model, optimizer, 'Best_Model', verbose=1
)

results['Best_MLP'] = metrics

print(f"\nWyniki najlepszego modelu:")
print(f"  F1-score: {metrics['f1']:.4f}")
print(f"  Recall: {metrics['recall']:.4f}")
print(f"  Precision: {metrics['precision']:.4f}")
print(f"  AUC-ROC: {metrics['auc']:.4f}")
print(f"  Accuracy: {metrics['accuracy']:.4f}")

# Zapisanie predykcji
np.save('../results/nn_best_pred.npy', y_pred)
np.save('../results/nn_best_proba.npy', y_pred_proba)
np.save('../results/nn_y_test.npy', y_test)

# ============================================================================
# PODSUMOWANIE I PORÓWNANIE Z RANDOM FOREST
# ============================================================================

print("\n" + "="*80)
print("PODSUMOWANIE WSZYSTKICH EKSPERYMENTÓW")
print("="*80)

# Tworzenie tabeli porównawczej
comparison_df = pd.DataFrame(results).T
comparison_df = comparison_df[['accuracy', 'precision', 'recall', 'f1', 'auc']]
comparison_df = comparison_df.round(4)

# Dodanie wyniku Random Forest dla porównania
rf_baseline = {
    'accuracy': 0.7333,
    'precision': 0.5484,
    'recall': 0.8947,
    'f1': 0.6800,
    'auc': 0.7689
}
comparison_df.loc['RF_Baseline'] = rf_baseline

print("\n" + comparison_df.to_string())

# Zapisanie wyników
comparison_df.to_csv('../results/neural_network_comparison.csv')
print("\n✓ Zapisano: results/neural_network_comparison.csv")

# Zapisanie szczegółowych wyników
import json
with open('../results/neural_network_details.json', 'w') as f:
    json.dump(results, f, indent=2)
print("✓ Zapisano: results/neural_network_details.json")

# Najlepszy model neuronowy
best_nn_name = comparison_df.drop('RF_Baseline')['f1'].idxmax()
best_nn_f1 = comparison_df.loc[best_nn_name, 'f1']

print(f"\n🏆 Najlepszy model neuronowy: {best_nn_name}")
print(f"   F1-score: {best_nn_f1:.4f}")
print(f"   Recall: {comparison_df.loc[best_nn_name, 'recall']:.4f}")
print(f"   AUC-ROC: {comparison_df.loc[best_nn_name, 'auc']:.4f}")

print(f"\n📊 Porównanie z Random Forest:")
print(f"   RF F1-score: {rf_baseline['f1']:.4f}")
print(f"   Najlepszy NN F1-score: {best_nn_f1:.4f}")
print(f"   Różnica: {(best_nn_f1 - rf_baseline['f1']):.4f} ({((best_nn_f1/rf_baseline['f1'] - 1)*100):.2f}%)")

print("\n" + "="*80)
print("EKSPERYMENTY ZAKOŃCZONE POMYŚLNIE")
print("="*80)
