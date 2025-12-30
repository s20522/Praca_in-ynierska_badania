"""
Generowanie Krzywych Uczenia dla Modeli MLP
============================================

Autor: Heart Failure Research Team
Data: 2025-12-30

Cel: Wygenerowanie profesjonalnych wykresów krzywych uczenia (training/validation loss i accuracy)
     dla kluczowych konfiguracji MLP, aby pokazać proces treningu i zidentyfikować przeuczenie.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras import layers, regularizers, callbacks
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Konfiguracja matplotlib
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 16
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'

# Ustawienie seed
np.random.seed(42)
tf.random.set_seed(42)

print("="*80)
print("GENEROWANIE KRZYWYCH UCZENIA DLA MODELI MLP")
print("="*80)

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

# Przygotowanie danych
X = df[['age', 'ejection_fraction', 'serum_creatinine']].values
y = df['DEATH_EVENT'].values

# Podział danych
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full
)

# Standaryzacja
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

print(f"\nRozmiary zbiorów:")
print(f"  Train: {X_train.shape[0]}")
print(f"  Validation: {X_val.shape[0]}")
print(f"  Test: {X_test.shape[0]}")

# Słownik do przechowywania historii
histories = {}

# Funkcja do budowy modelu
def build_model(architecture, activation='relu', dropout_rate=0.0, l2_reg=0.0):
    model = keras.Sequential()
    model.add(layers.Input(shape=(3,)))
    
    for units in architecture:
        if l2_reg > 0:
            model.add(layers.Dense(units, kernel_regularizer=regularizers.l2(l2_reg)))
        else:
            model.add(layers.Dense(units))
        
        if activation == 'relu':
            model.add(layers.Activation('relu'))
        elif activation == 'leaky_relu':
            model.add(layers.LeakyReLU(alpha=0.01))
        elif activation == 'elu':
            model.add(layers.ELU(alpha=1.0))
        
        if dropout_rate > 0:
            model.add(layers.Dropout(dropout_rate))
    
    model.add(layers.Dense(1, activation='sigmoid'))
    return model

# Funkcja do treningu
def train_model(model, optimizer, model_name):
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.AUC(name='auc')]
    )
    
    early_stop = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=0
    )
    
    history = model.fit(
        X_train_scaled, y_train,
        validation_data=(X_val_scaled, y_val),
        epochs=100,
        batch_size=16,
        callbacks=[early_stop],
        verbose=0
    )
    
    return history

# ============================================================================
# MODEL 1: Bez regularyzacji (pokazuje przeuczenie)
# ============================================================================

print("\n[1/5] Trenowanie: Model bez regularyzacji...")
model1 = build_model([128, 64], activation='relu', dropout_rate=0.0, l2_reg=0.0)
optimizer1 = keras.optimizers.Adam(learning_rate=0.001)
history1 = train_model(model1, optimizer1, "No Regularization")
histories['no_reg'] = history1.history
print(f"   Epochs: {len(history1.history['loss'])}")

# ============================================================================
# MODEL 2: Z Dropout (pokazuje efekt regularyzacji)
# ============================================================================

print("[2/5] Trenowanie: Model z Dropout=0.3...")
model2 = build_model([128, 64], activation='relu', dropout_rate=0.3, l2_reg=0.0)
optimizer2 = keras.optimizers.Adam(learning_rate=0.001)
history2 = train_model(model2, optimizer2, "Dropout 0.3")
histories['dropout'] = history2.history
print(f"   Epochs: {len(history2.history['loss'])}")

# ============================================================================
# MODEL 3: Z L2 (pokazuje efekt regularyzacji)
# ============================================================================

print("[3/5] Trenowanie: Model z L2=0.01...")
model3 = build_model([128, 64], activation='relu', dropout_rate=0.0, l2_reg=0.01)
optimizer3 = keras.optimizers.Adam(learning_rate=0.001)
history3 = train_model(model3, optimizer3, "L2 Regularization")
histories['l2'] = history3.history
print(f"   Epochs: {len(history3.history['loss'])}")

# ============================================================================
# MODEL 4: Najlepszy model (Dropout + L2)
# ============================================================================

print("[4/5] Trenowanie: Najlepszy model (Dropout + L2)...")
model4 = build_model([128, 64], activation='relu', dropout_rate=0.3, l2_reg=0.01)
optimizer4 = keras.optimizers.Adam(learning_rate=0.001)
history4 = train_model(model4, optimizer4, "Best Model")
histories['best'] = history4.history
print(f"   Epochs: {len(history4.history['loss'])}")

# ============================================================================
# MODEL 5: Płytka sieć (dla porównania)
# ============================================================================

print("[5/5] Trenowanie: Płytka sieć (128 neuronów)...")
model5 = build_model([128], activation='relu', dropout_rate=0.0, l2_reg=0.0)
optimizer5 = keras.optimizers.Adam(learning_rate=0.001)
history5 = train_model(model5, optimizer5, "Shallow Network")
histories['shallow'] = history5.history
print(f"   Epochs: {len(history5.history['loss'])}")

# Zapisanie historii
import json
with open('../results/mlp_training_histories.json', 'w') as f:
    # Konwersja numpy arrays na listy
    histories_serializable = {}
    for key, hist in histories.items():
        histories_serializable[key] = {k: [float(v) for v in vals] for k, vals in hist.items()}
    json.dump(histories_serializable, f, indent=2)

print("\n✓ Zapisano historie treningu: results/mlp_training_histories.json")

# ============================================================================
# WYKRES 1: Porównanie Loss (bez regularyzacji vs z regularyzacją)
# ============================================================================

print("\n[1/4] Generowanie: Porównanie Loss...")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Subplot 1: Model bez regularyzacji
ax1 = axes[0]
epochs1 = range(1, len(histories['no_reg']['loss']) + 1)
ax1.plot(epochs1, histories['no_reg']['loss'], 'b-', linewidth=2.5, label='Training Loss', marker='o', markersize=4)
ax1.plot(epochs1, histories['no_reg']['val_loss'], 'r-', linewidth=2.5, label='Validation Loss', marker='s', markersize=4)
ax1.set_xlabel('Epoka', fontsize=13, fontweight='bold')
ax1.set_ylabel('Binary Crossentropy Loss', fontsize=13, fontweight='bold')
ax1.set_title('Model BEZ Regularyzacji\n(Widoczne przeuczenie)', fontsize=14, fontweight='bold', pad=15)
ax1.legend(loc='upper right', fontsize=11, frameon=True, fancybox=True, shadow=True)
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# Oznaczenie punktu przeuczenia
if len(histories['no_reg']['val_loss']) > 10:
    min_val_loss_idx = np.argmin(histories['no_reg']['val_loss'])
    ax1.axvline(x=min_val_loss_idx+1, color='green', linestyle='--', linewidth=2, alpha=0.7)
    ax1.text(min_val_loss_idx+1, ax1.get_ylim()[1]*0.9, 'Optymalny\npunkt', 
            ha='center', fontsize=10, color='green', fontweight='bold')

# Subplot 2: Model z regularyzacją
ax2 = axes[1]
epochs2 = range(1, len(histories['best']['loss']) + 1)
ax2.plot(epochs2, histories['best']['loss'], 'b-', linewidth=2.5, label='Training Loss', marker='o', markersize=4)
ax2.plot(epochs2, histories['best']['val_loss'], 'r-', linewidth=2.5, label='Validation Loss', marker='s', markersize=4)
ax2.set_xlabel('Epoka', fontsize=13, fontweight='bold')
ax2.set_ylabel('Binary Crossentropy Loss', fontsize=13, fontweight='bold')
ax2.set_title('Najlepszy Model (Dropout + L2)\n(Regularyzacja działa!)', fontsize=14, fontweight='bold', pad=15)
ax2.legend(loc='upper right', fontsize=11, frameon=True, fancybox=True, shadow=True)
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('../results/nn_fig_08_learning_curves_loss.png')
print("   ✓ Zapisano: nn_fig_08_learning_curves_loss.png")
plt.close()

# ============================================================================
# WYKRES 2: Porównanie Accuracy
# ============================================================================

print("[2/4] Generowanie: Porównanie Accuracy...")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Subplot 1: Model bez regularyzacji
ax1 = axes[0]
epochs1 = range(1, len(histories['no_reg']['accuracy']) + 1)
ax1.plot(epochs1, histories['no_reg']['accuracy'], 'b-', linewidth=2.5, label='Training Accuracy', marker='o', markersize=4)
ax1.plot(epochs1, histories['no_reg']['val_accuracy'], 'r-', linewidth=2.5, label='Validation Accuracy', marker='s', markersize=4)
ax1.set_xlabel('Epoka', fontsize=13, fontweight='bold')
ax1.set_ylabel('Accuracy', fontsize=13, fontweight='bold')
ax1.set_title('Model BEZ Regularyzacji', fontsize=14, fontweight='bold', pad=15)
ax1.legend(loc='lower right', fontsize=11, frameon=True, fancybox=True, shadow=True)
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.set_ylim([0.4, 1.0])
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# Subplot 2: Model z regularyzacją
ax2 = axes[1]
epochs2 = range(1, len(histories['best']['accuracy']) + 1)
ax2.plot(epochs2, histories['best']['accuracy'], 'b-', linewidth=2.5, label='Training Accuracy', marker='o', markersize=4)
ax2.plot(epochs2, histories['best']['val_accuracy'], 'r-', linewidth=2.5, label='Validation Accuracy', marker='s', markersize=4)
ax2.set_xlabel('Epoka', fontsize=13, fontweight='bold')
ax2.set_ylabel('Accuracy', fontsize=13, fontweight='bold')
ax2.set_title('Najlepszy Model (Dropout + L2)', fontsize=14, fontweight='bold', pad=15)
ax2.legend(loc='lower right', fontsize=11, frameon=True, fancybox=True, shadow=True)
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.set_ylim([0.4, 1.0])
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('../results/nn_fig_09_learning_curves_accuracy.png')
print("   ✓ Zapisano: nn_fig_09_learning_curves_accuracy.png")
plt.close()

# ============================================================================
# WYKRES 3: Porównanie różnych technik regularyzacji
# ============================================================================

print("[3/4] Generowanie: Porównanie technik regularyzacji...")

fig, ax = plt.subplots(figsize=(14, 8))

models_to_compare = [
    ('no_reg', 'Bez regularyzacji', '#e74c3c', 'o'),
    ('dropout', 'Dropout=0.3', '#3498db', 's'),
    ('l2', 'L2=0.01', '#2ecc71', '^'),
    ('best', 'Dropout + L2', '#9b59b6', 'D')
]

for model_key, label, color, marker in models_to_compare:
    epochs = range(1, len(histories[model_key]['val_loss']) + 1)
    ax.plot(epochs, histories[model_key]['val_loss'], 
           linewidth=2.5, label=label, color=color, marker=marker, 
           markersize=6, markevery=max(1, len(epochs)//10))

ax.set_xlabel('Epoka', fontsize=13, fontweight='bold')
ax.set_ylabel('Validation Loss', fontsize=13, fontweight='bold')
ax.set_title('Wpływ różnych technik regularyzacji na Validation Loss', 
            fontsize=15, fontweight='bold', pad=20)
ax.legend(loc='upper right', fontsize=12, frameon=True, fancybox=True, shadow=True)
ax.grid(True, alpha=0.3, linestyle='--')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('../results/nn_fig_10_regularization_comparison.png')
print("   ✓ Zapisano: nn_fig_10_regularization_comparison.png")
plt.close()

# ============================================================================
# WYKRES 4: Porównanie architektur (Shallow vs Deep)
# ============================================================================

print("[4/4] Generowanie: Porównanie architektur...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Loss - Shallow
ax1 = axes[0, 0]
epochs_shallow = range(1, len(histories['shallow']['loss']) + 1)
ax1.plot(epochs_shallow, histories['shallow']['loss'], 'b-', linewidth=2.5, label='Training', marker='o', markersize=4)
ax1.plot(epochs_shallow, histories['shallow']['val_loss'], 'r-', linewidth=2.5, label='Validation', marker='s', markersize=4)
ax1.set_xlabel('Epoka', fontsize=12, fontweight='bold')
ax1.set_ylabel('Loss', fontsize=12, fontweight='bold')
ax1.set_title('Płytka sieć [128] - Loss', fontsize=13, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# Accuracy - Shallow
ax2 = axes[0, 1]
ax2.plot(epochs_shallow, histories['shallow']['accuracy'], 'b-', linewidth=2.5, label='Training', marker='o', markersize=4)
ax2.plot(epochs_shallow, histories['shallow']['val_accuracy'], 'r-', linewidth=2.5, label='Validation', marker='s', markersize=4)
ax2.set_xlabel('Epoka', fontsize=12, fontweight='bold')
ax2.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
ax2.set_title('Płytka sieć [128] - Accuracy', fontsize=13, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_ylim([0.4, 1.0])

# Loss - Deep
ax3 = axes[1, 0]
epochs_deep = range(1, len(histories['best']['loss']) + 1)
ax3.plot(epochs_deep, histories['best']['loss'], 'b-', linewidth=2.5, label='Training', marker='o', markersize=4)
ax3.plot(epochs_deep, histories['best']['val_loss'], 'r-', linewidth=2.5, label='Validation', marker='s', markersize=4)
ax3.set_xlabel('Epoka', fontsize=12, fontweight='bold')
ax3.set_ylabel('Loss', fontsize=12, fontweight='bold')
ax3.set_title('Głęboka sieć [128, 64] - Loss', fontsize=13, fontweight='bold')
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3)

# Accuracy - Deep
ax4 = axes[1, 1]
ax4.plot(epochs_deep, histories['best']['accuracy'], 'b-', linewidth=2.5, label='Training', marker='o', markersize=4)
ax4.plot(epochs_deep, histories['best']['val_accuracy'], 'r-', linewidth=2.5, label='Validation', marker='s', markersize=4)
ax4.set_xlabel('Epoka', fontsize=12, fontweight='bold')
ax4.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
ax4.set_title('Głęboka sieć [128, 64] - Accuracy', fontsize=13, fontweight='bold')
ax4.legend(fontsize=10)
ax4.grid(True, alpha=0.3)
ax4.set_ylim([0.4, 1.0])

plt.suptitle('Porównanie krzywych uczenia: Płytka vs Głęboka architektura', 
            fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('../results/nn_fig_11_architecture_learning_curves.png')
print("   ✓ Zapisano: nn_fig_11_architecture_learning_curves.png")
plt.close()

print("\n" + "="*80)
print("WSZYSTKIE KRZYWE UCZENIA ZOSTAŁY WYGENEROWANE POMYŚLNIE")
print("="*80)
print("\nWygenerowano 4 wykresy:")
print("  1. nn_fig_08_learning_curves_loss.png - Porównanie Loss (z i bez regularyzacji)")
print("  2. nn_fig_09_learning_curves_accuracy.png - Porównanie Accuracy")
print("  3. nn_fig_10_regularization_comparison.png - Wpływ różnych technik regularyzacji")
print("  4. nn_fig_11_architecture_learning_curves.png - Porównanie architektur")
