"""
Generowanie wykresów porównawczych - Eksperymenty z inżynierią cech
====================================================================

Autor: Heart Failure Research Team
Data: 2025-12-29

Cel: Wygenerowanie profesjonalnych wykresów porównujących różne podejścia
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
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

print("="*80)
print("GENEROWANIE WYKRESÓW PORÓWNAWCZYCH")
print("="*80)

# Wczytanie wyników
comparison_df = pd.read_csv('../results/feature_engineering_comparison.csv', index_col=0)
y_test = np.load('../results/fe_y_test.npy')

# Wczytanie predykcji
predictions = {
    'Baseline': {
        'pred': np.load('../results/fe_baseline_pred.npy'),
        'proba': np.load('../results/fe_baseline_proba.npy')
    },
    'Discretization': {
        'pred': np.load('../results/fe_discrete_pred.npy'),
        'proba': np.load('../results/fe_discrete_proba.npy')
    },
    'Interactions': {
        'pred': np.load('../results/fe_interact_pred.npy'),
        'proba': np.load('../results/fe_interact_proba.npy')
    },
    'MinMax': {
        'pred': np.load('../results/fe_minmax_pred.npy'),
        'proba': np.load('../results/fe_minmax_proba.npy')
    },
    'All_Features': {
        'pred': np.load('../results/fe_all_pred.npy'),
        'proba': np.load('../results/fe_all_proba.npy')
    }
}

# ============================================================================
# WYKRES 1: Porównanie wszystkich metryk
# ============================================================================

print("\n[1/5] Generowanie: Porównanie wszystkich metryk...")

metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
metrics_pl = {
    'accuracy': 'Accuracy\n(Dokładność)',
    'precision': 'Precision\n(Precyzja)',
    'recall': 'Recall\n(Czułość)',
    'f1': 'F1-score',
    'auc': 'AUC-ROC'
}

fig, axes = plt.subplots(1, 5, figsize=(20, 5))
colors = ['#3498db', '#e74c3c', '#2ecc71', '#e67e22', '#9b59b6']

for idx, metric in enumerate(metrics):
    ax = axes[idx]
    values = comparison_df[metric].values
    models = comparison_df.index.tolist()
    
    bars = ax.bar(range(len(models)), values, color=colors, alpha=0.8, 
                  edgecolor='black', linewidth=1.5)
    
    # Dodanie wartości na słupkach
    for i, (bar, value) in enumerate(zip(bars, values)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
               f'{value:.3f}',
               ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Wartość', fontsize=11, fontweight='bold')
    ax.set_title(metrics_pl[metric], fontsize=12, fontweight='bold', pad=10)
    ax.set_ylim([0, 1.1])
    ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.suptitle('Porównanie metryk dla różnych podejść do inżynierii cech', 
             fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('../results/fe_fig_01_metrics_comparison.png')
print("   ✓ Zapisano: fe_fig_01_metrics_comparison.png")
plt.close()

# ============================================================================
# WYKRES 2: Krzywe ROC dla wszystkich modeli
# ============================================================================

print("[2/5] Generowanie: Krzywe ROC...")

fig, ax = plt.subplots(figsize=(10, 8))

colors_roc = {
    'Baseline': '#3498db',
    'Discretization': '#e74c3c',
    'Interactions': '#2ecc71',
    'MinMax': '#e67e22',
    'All_Features': '#9b59b6'
}

for model_name, pred_data in predictions.items():
    fpr, tpr, _ = roc_curve(y_test, pred_data['proba'])
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, linewidth=2.5, label=f'{model_name} (AUC={roc_auc:.3f})',
            color=colors_roc[model_name])

ax.plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=2, 
        label='Losowy klasyfikator (AUC=0.500)')

ax.set_xlabel('False Positive Rate (FPR)', fontsize=13, fontweight='bold')
ax.set_ylabel('True Positive Rate (TPR)', fontsize=13, fontweight='bold')
ax.set_title('Porównanie krzywych ROC - wszystkie eksperymenty', 
             fontsize=15, fontweight='bold', pad=20)
ax.legend(loc='lower right', frameon=True, fancybox=True, shadow=True, fontsize=11)
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])

plt.tight_layout()
plt.savefig('../results/fe_fig_02_roc_comparison.png')
print("   ✓ Zapisano: fe_fig_02_roc_comparison.png")
plt.close()

# ============================================================================
# WYKRES 3: Porównanie F1-score i Recall (kluczowe metryki)
# ============================================================================

print("[3/5] Generowanie: Porównanie F1-score i Recall...")

fig, ax = plt.subplots(figsize=(12, 7))

x = np.arange(len(comparison_df))
width = 0.35

f1_values = comparison_df['f1'].values
recall_values = comparison_df['recall'].values

bars1 = ax.bar(x - width/2, f1_values, width, label='F1-score', 
               color='#3498db', alpha=0.8, edgecolor='black', linewidth=1.5)
bars2 = ax.bar(x + width/2, recall_values, width, label='Recall (Czułość)', 
               color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=1.5)

# Dodanie wartości na słupkach
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
               f'{height:.3f}',
               ha='center', va='bottom', fontsize=10, fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(comparison_df.index, fontsize=11)
ax.set_ylabel('Wartość metryki', fontsize=13, fontweight='bold')
ax.set_title('Porównanie F1-score i Recall - kluczowe metryki w medycynie', 
             fontsize=15, fontweight='bold', pad=20)
ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True, fontsize=12)
ax.set_ylim([0, 1.1])
ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('../results/fe_fig_03_f1_recall_comparison.png')
print("   ✓ Zapisano: fe_fig_03_f1_recall_comparison.png")
plt.close()

# ============================================================================
# WYKRES 4: Trade-off Precision vs Recall
# ============================================================================

print("[4/5] Generowanie: Trade-off Precision vs Recall...")

fig, ax = plt.subplots(figsize=(10, 8))

precision_values = comparison_df['precision'].values
recall_values = comparison_df['recall'].values
model_names = comparison_df.index.tolist()

# Scatter plot
for i, (prec, rec, name) in enumerate(zip(precision_values, recall_values, model_names)):
    ax.scatter(rec, prec, s=300, color=colors[i], alpha=0.7, 
              edgecolor='black', linewidth=2, label=name)
    ax.annotate(name, (rec, prec), xytext=(10, 10), textcoords='offset points',
               fontsize=10, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.5', facecolor=colors[i], alpha=0.3))

# Linie pomocnicze
ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5)
ax.axvline(x=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5)

ax.set_xlabel('Recall (Czułość)', fontsize=13, fontweight='bold')
ax.set_ylabel('Precision (Precyzja)', fontsize=13, fontweight='bold')
ax.set_title('Trade-off między Precision i Recall', 
             fontsize=15, fontweight='bold', pad=20)
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_xlim([0.4, 1.0])
ax.set_ylim([0.3, 0.7])

# Dodanie strzałek wskazujących kierunki optymalne
ax.annotate('', xy=(0.95, 0.65), xytext=(0.75, 0.45),
           arrowprops=dict(arrowstyle='->', lw=2, color='green', alpha=0.5))
ax.text(0.85, 0.55, 'Idealny kierunek', fontsize=11, color='green', 
        fontweight='bold', rotation=30)

plt.tight_layout()
plt.savefig('../results/fe_fig_04_precision_recall_tradeoff.png')
print("   ✓ Zapisano: fe_fig_04_precision_recall_tradeoff.png")
plt.close()

# ============================================================================
# WYKRES 5: Tabela podsumowująca
# ============================================================================

print("[5/5] Generowanie: Tabela podsumowująca...")

fig, ax = plt.subplots(figsize=(14, 6))
ax.axis('tight')
ax.axis('off')

# Przygotowanie danych do tabeli
table_data = []
for idx, row in comparison_df.iterrows():
    table_data.append([
        idx,
        f"{row['accuracy']:.4f}",
        f"{row['precision']:.4f}",
        f"{row['recall']:.4f}",
        f"{row['f1']:.4f}",
        f"{row['auc']:.4f}",
        int(row['n_features'])
    ])

# Nagłówki
headers = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1-score', 'AUC-ROC', 'Liczba cech']

# Tworzenie tabeli
table = ax.table(cellText=table_data, colLabels=headers, cellLoc='center',
                loc='center', bbox=[0, 0, 1, 1])

# Stylizacja
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2.5)

# Kolorowanie nagłówków
for i in range(len(headers)):
    cell = table[(0, i)]
    cell.set_facecolor('#3498db')
    cell.set_text_props(weight='bold', color='white', fontsize=12)

# Kolorowanie wierszy
for i in range(1, len(table_data) + 1):
    for j in range(len(headers)):
        cell = table[(i, j)]
        if i % 2 == 0:
            cell.set_facecolor('#ecf0f1')
        else:
            cell.set_facecolor('white')
        
        # Pogrubienie najlepszych wyników
        if j in [1, 2, 3, 4, 5]:  # metryki
            value = float(table_data[i-1][j])
            col_values = [float(row[j]) for row in table_data]
            if value == max(col_values):
                cell.set_text_props(weight='bold', color='#27ae60')

plt.title('Podsumowanie wyników - wszystkie eksperymenty z inżynierią cech', 
          fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('../results/fe_fig_05_summary_table.png')
print("   ✓ Zapisano: fe_fig_05_summary_table.png")
plt.close()

print("\n" + "="*80)
print("WSZYSTKIE WYKRESY ZOSTAŁY WYGENEROWANE POMYŚLNIE")
print("="*80)
print("\nWygenerowano 5 wykresów:")
print("  1. fe_fig_01_metrics_comparison.png")
print("  2. fe_fig_02_roc_comparison.png")
print("  3. fe_fig_03_f1_recall_comparison.png")
print("  4. fe_fig_04_precision_recall_tradeoff.png")
print("  5. fe_fig_05_summary_table.png")
