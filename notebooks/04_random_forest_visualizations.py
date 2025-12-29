"""
Generowanie wykresów dla modelu Random Forest
==============================================

Autor: Heart Failure Research Team
Data: 2025-12-29

Cel: Wygenerowanie profesjonalnych wykresów do pracy inżynierskiej
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, auc
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

COLOR_PRIMARY = '#3498db'
COLOR_SECONDARY = '#e74c3c'

print("="*80)
print("GENEROWANIE WYKRESÓW - RANDOM FOREST")
print("="*80)

# Wczytanie danych
y_test = np.load('../results/rf_y_test.npy')
y_pred = np.load('../results/rf_y_pred.npy')
y_pred_proba = np.load('../results/rf_y_pred_proba.npy')
feature_importances = np.load('../results/rf_feature_importances.npy')
feature_names = ['age', 'ejection_fraction', 'serum_creatinine']

# ============================================================================
# WYKRES 1: Macierz pomyłek (Confusion Matrix)
# ============================================================================

print("\n[1/5] Generowanie: Macierz pomyłek...")

cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots(figsize=(10, 8))

# Heatmapa
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
            xticklabels=['Przeżył (0)', 'Zmarł (1)'],
            yticklabels=['Przeżył (0)', 'Zmarł (1)'],
            annot_kws={'fontsize': 16, 'fontweight': 'bold'},
            ax=ax, vmin=0, linewidths=2, linecolor='black')

ax.set_xlabel('Predykcja', fontsize=13, fontweight='bold')
ax.set_ylabel('Prawdziwa wartość', fontsize=13, fontweight='bold')
ax.set_title('Macierz pomyłek (Confusion Matrix) - Random Forest', 
             fontsize=15, fontweight='bold', pad=20)

# Dodanie etykiet TN, FP, FN, TP
labels = [['TN\n(True Negative)', 'FP\n(False Positive)'],
          ['FN\n(False Negative)', 'TP\n(True Positive)']]

for i in range(2):
    for j in range(2):
        ax.text(j+0.5, i+0.7, labels[i][j], 
               ha='center', va='center', fontsize=10, color='darkred', style='italic')

plt.tight_layout()
plt.savefig('../results/rf_fig_01_confusion_matrix.png')
print("   ✓ Zapisano: rf_fig_01_confusion_matrix.png")
plt.close()

# ============================================================================
# WYKRES 2: Krzywa ROC (ROC Curve)
# ============================================================================

print("[2/5] Generowanie: Krzywa ROC...")

fpr, tpr, thresholds_roc = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

fig, ax = plt.subplots(figsize=(10, 8))

ax.plot(fpr, tpr, color=COLOR_SECONDARY, linewidth=3, 
        label=f'Random Forest (AUC = {roc_auc:.4f})')
ax.plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=2, 
        label='Losowy klasyfikator (AUC = 0.5000)')

ax.set_xlabel('False Positive Rate (FPR)', fontsize=13, fontweight='bold')
ax.set_ylabel('True Positive Rate (TPR)', fontsize=13, fontweight='bold')
ax.set_title('Krzywa ROC (Receiver Operating Characteristic)', 
             fontsize=15, fontweight='bold', pad=20)
ax.legend(loc='lower right', frameon=True, fancybox=True, shadow=True, fontsize=12)
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])

plt.tight_layout()
plt.savefig('../results/rf_fig_02_roc_curve.png')
print("   ✓ Zapisano: rf_fig_02_roc_curve.png")
plt.close()

# ============================================================================
# WYKRES 3: Krzywa Precision-Recall
# ============================================================================

print("[3/5] Generowanie: Krzywa Precision-Recall...")

precision, recall, thresholds_pr = precision_recall_curve(y_test, y_pred_proba)
pr_auc = auc(recall, precision)

fig, ax = plt.subplots(figsize=(10, 8))

ax.plot(recall, precision, color=COLOR_PRIMARY, linewidth=3, 
        label=f'Random Forest (AUC = {pr_auc:.4f})')

# Baseline (proporcja klasy pozytywnej)
baseline = np.sum(y_test) / len(y_test)
ax.plot([0, 1], [baseline, baseline], color='gray', linestyle='--', linewidth=2, 
        label=f'Baseline (proporcja kl. poz. = {baseline:.4f})')

ax.set_xlabel('Recall (Czułość)', fontsize=13, fontweight='bold')
ax.set_ylabel('Precision (Precyzja)', fontsize=13, fontweight='bold')
ax.set_title('Krzywa Precision-Recall', 
             fontsize=15, fontweight='bold', pad=20)
ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True, fontsize=12)
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])

plt.tight_layout()
plt.savefig('../results/rf_fig_03_precision_recall_curve.png')
print("   ✓ Zapisano: rf_fig_03_precision_recall_curve.png")
plt.close()

# ============================================================================
# WYKRES 4: Ważność cech (Feature Importance)
# ============================================================================

print("[4/5] Generowanie: Ważność cech...")

feature_names_pl = {
    'age': 'Wiek',
    'ejection_fraction': 'Frakcja wyrzutowa',
    'serum_creatinine': 'Kreatynina w surowicy'
}

labels_pl = [feature_names_pl[name] for name in feature_names]

# Sortowanie
indices = np.argsort(feature_importances)[::-1]

fig, ax = plt.subplots(figsize=(10, 6))

colors = [COLOR_SECONDARY, COLOR_PRIMARY, '#2ecc71']
bars = ax.bar(range(len(feature_importances)), feature_importances[indices], 
              color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

# Dodanie wartości na słupkach
for i, (bar, importance) in enumerate(zip(bars, feature_importances[indices])):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
           f'{importance:.4f}\n({importance*100:.2f}%)',
           ha='center', va='bottom', fontsize=11, fontweight='bold')

ax.set_xticks(range(len(feature_importances)))
ax.set_xticklabels([labels_pl[i] for i in indices], fontsize=12)
ax.set_ylabel('Ważność cechy (Feature Importance)', fontsize=13, fontweight='bold')
ax.set_title('Ważność cech w modelu Random Forest', 
             fontsize=15, fontweight='bold', pad=20)
ax.set_ylim([0, max(feature_importances) * 1.2])
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('../results/rf_fig_04_feature_importance.png')
print("   ✓ Zapisano: rf_fig_04_feature_importance.png")
plt.close()

# ============================================================================
# WYKRES 5: Podsumowanie metryk
# ============================================================================

print("[5/5] Generowanie: Podsumowanie metryk...")

# Wczytanie wyników z pliku
with open('../results/random_forest_results.txt', 'r', encoding='utf-8') as f:
    content = f.read()

# Parsowanie metryk
import re
accuracy = float(re.search(r'Accuracy:\s+([\d.]+)', content).group(1))
precision = float(re.search(r'Precision:\s+([\d.]+)', content).group(1))
recall = float(re.search(r'Recall:\s+([\d.]+)', content).group(1))
f1 = float(re.search(r'F1-score:\s+([\d.]+)', content).group(1))
auc_roc = float(re.search(r'AUC-ROC:\s+([\d.]+)', content).group(1))

metrics = {
    'Accuracy\n(Dokładność)': accuracy,
    'Precision\n(Precyzja)': precision,
    'Recall\n(Czułość)': recall,
    'F1-score': f1,
    'AUC-ROC': auc_roc
}

fig, ax = plt.subplots(figsize=(12, 7))

x_pos = np.arange(len(metrics))
values = list(metrics.values())
colors_metrics = [COLOR_PRIMARY, COLOR_SECONDARY, '#2ecc71', '#e67e22', '#9b59b6']

bars = ax.bar(x_pos, values, color=colors_metrics, alpha=0.8, 
              edgecolor='black', linewidth=1.5)

# Dodanie wartości na słupkach
for i, (bar, value) in enumerate(zip(bars, values)):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
           f'{value:.4f}\n({value*100:.2f}%)',
           ha='center', va='bottom', fontsize=11, fontweight='bold')

ax.set_xticks(x_pos)
ax.set_xticklabels(list(metrics.keys()), fontsize=11)
ax.set_ylabel('Wartość metryki', fontsize=13, fontweight='bold')
ax.set_title('Podsumowanie metryk - Random Forest (zbiór testowy)', 
             fontsize=15, fontweight='bold', pad=20)
ax.set_ylim([0, 1.1])
ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('../results/rf_fig_05_metrics_summary.png')
print("   ✓ Zapisano: rf_fig_05_metrics_summary.png")
plt.close()

print("\n" + "="*80)
print("WSZYSTKIE WYKRESY ZOSTAŁY WYGENEROWANE POMYŚLNIE")
print("="*80)
print("\nWygenerowano 5 wykresów:")
print("  1. rf_fig_01_confusion_matrix.png")
print("  2. rf_fig_02_roc_curve.png")
print("  3. rf_fig_03_precision_recall_curve.png")
print("  4. rf_fig_04_feature_importance.png")
print("  5. rf_fig_05_metrics_summary.png")
