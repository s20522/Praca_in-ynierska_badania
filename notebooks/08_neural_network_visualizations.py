"""
Generowanie wykresów porównawczych - Eksperymenty z Sieciami Neuronowymi
=========================================================================

Autor: Heart Failure Research Team
Data: 2025-12-29

Cel: Wygenerowanie profesjonalnych wykresów porównujących różne konfiguracje MLP
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
print("GENEROWANIE WYKRESÓW - EKSPERYMENTY Z SIECIAMI NEURONOWYMI")
print("="*80)

# Wczytanie wyników
comparison_df = pd.read_csv('../results/neural_network_comparison.csv', index_col=0)
y_test = np.load('../results/nn_y_test.npy')

# ============================================================================
# WYKRES 1: Porównanie architektur
# ============================================================================

print("\n[1/7] Generowanie: Porównanie architektur...")

arch_models = [col for col in comparison_df.index if col.startswith('Arch_')]
arch_df = comparison_df.loc[arch_models]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
metrics = ['f1', 'recall', 'precision', 'auc']
titles = ['F1-score', 'Recall (Czułość)', 'Precision (Precyzja)', 'AUC-ROC']

for idx, (metric, title) in enumerate(zip(metrics, titles)):
    ax = axes[idx//2, idx%2]
    values = arch_df[metric].values
    labels = [name.replace('Arch_', '') for name in arch_df.index]
    
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(values)))
    bars = ax.barh(range(len(labels)), values, color=colors, alpha=0.8,
                   edgecolor='black', linewidth=1.5)
    
    # Dodanie wartości
    for i, (bar, value) in enumerate(zip(bars, values)):
        width = bar.get_width()
        ax.text(width + 0.01, bar.get_y() + bar.get_height()/2,
               f'{value:.3f}',
               ha='left', va='center', fontsize=9, fontweight='bold')
    
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel('Wartość', fontsize=11, fontweight='bold')
    ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
    ax.set_xlim([0, 1.0])
    ax.axvline(x=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.suptitle('Porównanie różnych architektur sieci neuronowych', 
             fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('../results/nn_fig_01_architecture_comparison.png')
print("   ✓ Zapisano: nn_fig_01_architecture_comparison.png")
plt.close()

# ============================================================================
# WYKRES 2: Porównanie funkcji aktywacji
# ============================================================================

print("[2/7] Generowanie: Porównanie funkcji aktywacji...")

act_models = [col for col in comparison_df.index if col.startswith('Activation_')]
act_df = comparison_df.loc[act_models]

fig, ax = plt.subplots(figsize=(12, 7))

x = np.arange(len(act_df))
width = 0.2

metrics_to_plot = ['f1', 'recall', 'precision', 'auc']
colors_map = {'f1': '#3498db', 'recall': '#e74c3c', 'precision': '#2ecc71', 'auc': '#e67e22'}
labels_map = {'f1': 'F1-score', 'recall': 'Recall', 'precision': 'Precision', 'auc': 'AUC-ROC'}

for i, metric in enumerate(metrics_to_plot):
    values = act_df[metric].values
    offset = width * (i - 1.5)
    bars = ax.bar(x + offset, values, width, label=labels_map[metric],
                  color=colors_map[metric], alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Dodanie wartości na słupkach
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
               f'{height:.3f}',
               ha='center', va='bottom', fontsize=8, fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels([name.replace('Activation_', '').upper() for name in act_df.index], fontsize=11)
ax.set_ylabel('Wartość metryki', fontsize=13, fontweight='bold')
ax.set_title('Porównanie funkcji aktywacji', fontsize=15, fontweight='bold', pad=20)
ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True, fontsize=11)
ax.set_ylim([0, 1.0])
ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('../results/nn_fig_02_activation_comparison.png')
print("   ✓ Zapisano: nn_fig_02_activation_comparison.png")
plt.close()

# ============================================================================
# WYKRES 3: Wpływ Dropout
# ============================================================================

print("[3/7] Generowanie: Wpływ Dropout...")

dropout_models = [col for col in comparison_df.index if col.startswith('Dropout_')]
dropout_df = comparison_df.loc[dropout_models]
dropout_rates = [float(name.replace('Dropout_', '')) for name in dropout_df.index]

fig, ax = plt.subplots(figsize=(12, 7))

for metric, label, color, marker in [('f1', 'F1-score', '#3498db', 'o'),
                                      ('recall', 'Recall', '#e74c3c', 's'),
                                      ('auc', 'AUC-ROC', '#2ecc71', '^')]:
    values = dropout_df[metric].values
    ax.plot(dropout_rates, values, marker=marker, markersize=10, linewidth=2.5,
           label=label, color=color, alpha=0.8)

ax.set_xlabel('Dropout Rate', fontsize=13, fontweight='bold')
ax.set_ylabel('Wartość metryki', fontsize=13, fontweight='bold')
ax.set_title('Wpływ współczynnika Dropout na wydajność modelu', 
             fontsize=15, fontweight='bold', pad=20)
ax.legend(loc='best', frameon=True, fancybox=True, shadow=True, fontsize=12)
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_xticks(dropout_rates)
ax.set_ylim([0.4, 0.9])

plt.tight_layout()
plt.savefig('../results/nn_fig_03_dropout_effect.png')
print("   ✓ Zapisano: nn_fig_03_dropout_effect.png")
plt.close()

# ============================================================================
# WYKRES 4: Wpływ regularyzacji L2
# ============================================================================

print("[4/7] Generowanie: Wpływ regularyzacji L2...")

l2_models = [col for col in comparison_df.index if col.startswith('L2_')]
l2_df = comparison_df.loc[l2_models]
l2_values = [float(name.replace('L2_', '')) for name in l2_df.index]

fig, ax = plt.subplots(figsize=(12, 7))

for metric, label, color, marker in [('f1', 'F1-score', '#3498db', 'o'),
                                      ('recall', 'Recall', '#e74c3c', 's'),
                                      ('precision', 'Precision', '#9b59b6', 'd')]:
    values = l2_df[metric].values
    ax.plot(l2_values, values, marker=marker, markersize=10, linewidth=2.5,
           label=label, color=color, alpha=0.8)

ax.set_xlabel('Współczynnik regularyzacji L2', fontsize=13, fontweight='bold')
ax.set_ylabel('Wartość metryki', fontsize=13, fontweight='bold')
ax.set_title('Wpływ regularyzacji L2 na wydajność modelu', 
             fontsize=15, fontweight='bold', pad=20)
ax.legend(loc='best', frameon=True, fancybox=True, shadow=True, fontsize=12)
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_xscale('log')
ax.set_ylim([0.2, 0.9])

plt.tight_layout()
plt.savefig('../results/nn_fig_04_l2_effect.png')
print("   ✓ Zapisano: nn_fig_04_l2_effect.png")
plt.close()

# ============================================================================
# WYKRES 5: Porównanie optymalizatorów
# ============================================================================

print("[5/7] Generowanie: Porównanie optymalizatorów...")

opt_models = [col for col in comparison_df.index if col.startswith('Optimizer_')]
opt_df = comparison_df.loc[opt_models]

fig, ax = plt.subplots(figsize=(12, 7))

x = np.arange(len(opt_df))
width = 0.2

for i, metric in enumerate(['f1', 'recall', 'precision', 'auc']):
    values = opt_df[metric].values
    offset = width * (i - 1.5)
    bars = ax.bar(x + offset, values, width, label=labels_map[metric],
                  color=colors_map[metric], alpha=0.8, edgecolor='black', linewidth=1.5)
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
               f'{height:.3f}',
               ha='center', va='bottom', fontsize=8, fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels([name.replace('Optimizer_', '') for name in opt_df.index], fontsize=11)
ax.set_ylabel('Wartość metryki', fontsize=13, fontweight='bold')
ax.set_title('Porównanie optymalizatorów', fontsize=15, fontweight='bold', pad=20)
ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True, fontsize=11)
ax.set_ylim([0, 1.0])
ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('../results/nn_fig_05_optimizer_comparison.png')
print("   ✓ Zapisano: nn_fig_05_optimizer_comparison.png")
plt.close()

# ============================================================================
# WYKRES 6: MLP vs Random Forest - Porównanie kluczowych metryk
# ============================================================================

print("[6/7] Generowanie: MLP vs Random Forest...")

# Wybór najlepszego modelu MLP
best_nn_models = ['Best_MLP', 'Optimizer_Adam', 'Dropout_0.5', 'L2_0.001']
comparison_subset = comparison_df.loc[best_nn_models + ['RF_Baseline']]

fig, ax = plt.subplots(figsize=(14, 8))

x = np.arange(len(comparison_subset))
width = 0.18

for i, metric in enumerate(['f1', 'recall', 'precision', 'auc']):
    values = comparison_subset[metric].values
    offset = width * (i - 1.5)
    bars = ax.bar(x + offset, values, width, label=labels_map[metric],
                  color=colors_map[metric], alpha=0.8, edgecolor='black', linewidth=1.5)
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
               f'{height:.3f}',
               ha='center', va='bottom', fontsize=9, fontweight='bold')

ax.set_xticks(x)
labels = ['Best MLP', 'Adam', 'Dropout=0.5', 'L2=0.001', 'Random Forest']
ax.set_xticklabels(labels, fontsize=11, fontweight='bold')
ax.set_ylabel('Wartość metryki', fontsize=13, fontweight='bold')
ax.set_title('Porównanie najlepszych modeli MLP z Random Forest', 
             fontsize=15, fontweight='bold', pad=20)
ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True, fontsize=12)
ax.set_ylim([0, 1.0])
ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Podświetlenie RF
ax.axvspan(3.5, 4.5, alpha=0.1, color='green')

plt.tight_layout()
plt.savefig('../results/nn_fig_06_mlp_vs_rf.png')
print("   ✓ Zapisano: nn_fig_06_mlp_vs_rf.png")
plt.close()

# ============================================================================
# WYKRES 7: Tabela podsumowująca
# ============================================================================

print("[7/7] Generowanie: Tabela podsumowująca...")

fig, ax = plt.subplots(figsize=(16, 10))
ax.axis('tight')
ax.axis('off')

# Top 10 modeli według F1-score
top_models = comparison_df.nlargest(10, 'f1')

# Przygotowanie danych do tabeli
table_data = []
for idx, row in top_models.iterrows():
    table_data.append([
        idx,
        f"{row['accuracy']:.4f}",
        f"{row['precision']:.4f}",
        f"{row['recall']:.4f}",
        f"{row['f1']:.4f}",
        f"{row['auc']:.4f}"
    ])

# Nagłówki
headers = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1-score', 'AUC-ROC']

# Tworzenie tabeli
table = ax.table(cellText=table_data, colLabels=headers, cellLoc='center',
                loc='center', bbox=[0, 0, 1, 1])

# Stylizacja
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2.5)

# Kolorowanie nagłówków
for i in range(len(headers)):
    cell = table[(0, i)]
    cell.set_facecolor('#3498db')
    cell.set_text_props(weight='bold', color='white', fontsize=11)

# Kolorowanie wierszy
for i in range(1, len(table_data) + 1):
    for j in range(len(headers)):
        cell = table[(i, j)]
        
        # Podświetlenie RF
        if 'RF_Baseline' in table_data[i-1][0]:
            cell.set_facecolor('#d5f4e6')
        elif i % 2 == 0:
            cell.set_facecolor('#ecf0f1')
        else:
            cell.set_facecolor('white')
        
        # Pogrubienie najlepszych wyników
        if j in [1, 2, 3, 4, 5]:  # metryki
            value = float(table_data[i-1][j])
            col_values = [float(row[j]) for row in table_data]
            if value == max(col_values):
                cell.set_text_props(weight='bold', color='#27ae60')

plt.title('Top 10 modeli - Porównanie wszystkich eksperymentów', 
          fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('../results/nn_fig_07_summary_table.png')
print("   ✓ Zapisano: nn_fig_07_summary_table.png")
plt.close()

print("\n" + "="*80)
print("WSZYSTKIE WYKRESY ZOSTAŁY WYGENEROWANE POMYŚLNIE")
print("="*80)
print("\nWygenerowano 7 wykresów:")
print("  1. nn_fig_01_architecture_comparison.png")
print("  2. nn_fig_02_activation_comparison.png")
print("  3. nn_fig_03_dropout_effect.png")
print("  4. nn_fig_04_l2_effect.png")
print("  5. nn_fig_05_optimizer_comparison.png")
print("  6. nn_fig_06_mlp_vs_rf.png")
print("  7. nn_fig_07_summary_table.png")
