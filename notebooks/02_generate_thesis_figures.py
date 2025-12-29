"""
Generowanie profesjonalnych wykresów do pracy inżynierskiej
============================================================

Autor: Heart Failure Research Team
Data: 2025-12-29

Cel: Wygenerowanie wykresów wysokiej jakości z polskimi opisami,
     odpowiednich do umieszczenia w pracy inżynierskiej.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Konfiguracja matplotlib dla polskich znaków i wysokiej jakości
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
plt.rcParams['savefig.pad_inches'] = 0.1

# Paleta kolorów
COLOR_SURVIVED = '#2ecc71'  # Zielony
COLOR_DIED = '#e74c3c'      # Czerwony
COLOR_PRIMARY = '#3498db'   # Niebieski
COLOR_SECONDARY = '#e67e22' # Pomarańczowy

print("="*80)
print("GENEROWANIE WYKRESÓW DO PRACY INŻYNIERSKIEJ")
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

# Polskie nazwy cech
feature_names_pl = {
    'age': 'Wiek [lata]',
    'ejection_fraction': 'Frakcja wyrzutowa [%]',
    'serum_creatinine': 'Kreatynina w surowicy [mg/dL]',
    'serum_sodium': 'Sód w surowicy [mEq/L]',
    'platelets': 'Płytki krwi [×10³/μL]',
    'creatinine_phosphokinase': 'Kinaza kreatynowa [mcg/L]',
    'time': 'Czas obserwacji [dni]',
    'sex': 'Płeć',
    'smoking': 'Palenie',
    'diabetes': 'Cukrzyca',
    'high_blood_pressure': 'Nadciśnienie',
    'anaemia': 'Anemia'
}

# ============================================================================
# WYKRES 1: Rozkład zmiennej celu z dokładnymi liczbami
# ============================================================================

print("\n[1/10] Generowanie: Rozkład zmiennej celu...")

fig, ax = plt.subplots(figsize=(10, 6))

death_counts = df['DEATH_EVENT'].value_counts()
labels = ['Przeżyli', 'Zmarli']
colors = [COLOR_SURVIVED, COLOR_DIED]

bars = ax.bar(labels, death_counts.values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

# Dodanie wartości na słupkach
for i, (bar, count) in enumerate(zip(bars, death_counts.values)):
    height = bar.get_height()
    percentage = (count / len(df)) * 100
    ax.text(bar.get_x() + bar.get_width()/2., height + 3,
            f'{count}\n({percentage:.1f}%)',
            ha='center', va='bottom', fontsize=12, fontweight='bold')

ax.set_ylabel('Liczba pacjentów', fontsize=13, fontweight='bold')
ax.set_title('Rozkład zmiennej celu (DEATH_EVENT) w zbiorze danych', 
             fontsize=15, fontweight='bold', pad=20)
ax.set_ylim(0, max(death_counts.values) * 1.15)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('../results/thesis_fig_01_target_distribution.png')
print("   ✓ Zapisano: thesis_fig_01_target_distribution.png")
plt.close()

# ============================================================================
# WYKRES 2: Macierz korelacji z DEATH_EVENT (top cechy)
# ============================================================================

print("[2/10] Generowanie: Korelacje z DEATH_EVENT...")

fig, ax = plt.subplots(figsize=(10, 8))

# Korelacje z DEATH_EVENT (bez samej DEATH_EVENT i time)
correlation_matrix = df.corr()
death_correlations = correlation_matrix['DEATH_EVENT'].drop(['DEATH_EVENT', 'time']).sort_values()

# Kolory: dodatnie = czerwony, ujemne = zielony
colors_corr = [COLOR_DIED if x > 0 else COLOR_SURVIVED for x in death_correlations.values]

bars = ax.barh(range(len(death_correlations)), death_correlations.values, 
               color=colors_corr, alpha=0.8, edgecolor='black', linewidth=1)

# Polskie nazwy
y_labels = [feature_names_pl.get(feat, feat) for feat in death_correlations.index]
ax.set_yticks(range(len(death_correlations)))
ax.set_yticklabels(y_labels)

# Dodanie wartości na końcach słupków
for i, (bar, val) in enumerate(zip(bars, death_correlations.values)):
    x_pos = val + (0.01 if val > 0 else -0.01)
    ha = 'left' if val > 0 else 'right'
    ax.text(x_pos, i, f'{val:.3f}', va='center', ha=ha, fontsize=9, fontweight='bold')

ax.set_xlabel('Współczynnik korelacji Pearsona', fontsize=13, fontweight='bold')
ax.set_title('Korelacje cech z DEATH_EVENT\n(bez cechy "time" - target leakage)', 
             fontsize=15, fontweight='bold', pad=20)
ax.axvline(0, color='black', linewidth=1.5, linestyle='-')
ax.grid(axis='x', alpha=0.3, linestyle='--')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Legenda
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=COLOR_DIED, alpha=0.8, edgecolor='black', label='Korelacja dodatnia (↑ ryzyko)'),
    Patch(facecolor=COLOR_SURVIVED, alpha=0.8, edgecolor='black', label='Korelacja ujemna (↓ ryzyko)')
]
ax.legend(handles=legend_elements, loc='lower right', frameon=True, fancybox=True, shadow=True)

plt.tight_layout()
plt.savefig('../results/thesis_fig_02_correlations.png')
print("   ✓ Zapisano: thesis_fig_02_correlations.png")
plt.close()

# ============================================================================
# WYKRES 3: Porównanie kluczowych cech (przeżyli vs zmarli) - 4 panele
# ============================================================================

print("[3/10] Generowanie: Porównanie kluczowych cech...")

survived = df[df['DEATH_EVENT'] == 0]
died = df[df['DEATH_EVENT'] == 1]

key_features = ['ejection_fraction', 'serum_creatinine', 'age', 'serum_sodium']

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.ravel()

for idx, feature in enumerate(key_features):
    ax = axes[idx]
    
    # Wykresy pudełkowe
    data_to_plot = [survived[feature], died[feature]]
    bp = ax.boxplot(data_to_plot, labels=['Przeżyli', 'Zmarli'],
                    patch_artist=True, widths=0.6,
                    boxprops=dict(linewidth=1.5),
                    whiskerprops=dict(linewidth=1.5),
                    capprops=dict(linewidth=1.5),
                    medianprops=dict(linewidth=2, color='darkred'))
    
    # Kolorowanie
    colors = [COLOR_SURVIVED, COLOR_DIED]
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # Dodanie punktów danych (violin plot style)
    positions = [1, 2]
    for i, data in enumerate(data_to_plot):
        y = data
        x = np.random.normal(positions[i], 0.04, size=len(y))
        ax.scatter(x, y, alpha=0.3, s=20, color=colors[i], edgecolors='black', linewidth=0.5)
    
    # Test t-Studenta
    t_stat, p_value = stats.ttest_ind(survived[feature], died[feature])
    
    # Statystyki
    mean_survived = survived[feature].mean()
    mean_died = died[feature].mean()
    
    ax.set_ylabel(feature_names_pl[feature], fontsize=12, fontweight='bold')
    ax.set_title(f'{feature_names_pl[feature]}', fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Adnotacja z p-value i średnimi
    significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
    textstr = f'p-value: {p_value:.4f} {significance}\n'
    textstr += f'Średnia (przeżyli): {mean_survived:.2f}\n'
    textstr += f'Średnia (zmarli): {mean_died:.2f}'
    
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, 
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

fig.suptitle('Porównanie kluczowych cech klinicznych: Przeżyli vs Zmarli', 
             fontsize=16, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig('../results/thesis_fig_03_key_features_comparison.png')
print("   ✓ Zapisano: thesis_fig_03_key_features_comparison.png")
plt.close()

# ============================================================================
# WYKRES 4: Rozkłady najważniejszych cech numerycznych
# ============================================================================

print("[4/10] Generowanie: Rozkłady cech numerycznych...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.ravel()

for idx, feature in enumerate(key_features):
    ax = axes[idx]
    
    # Histogram dla obu grup
    ax.hist(survived[feature], bins=20, alpha=0.6, color=COLOR_SURVIVED, 
            edgecolor='black', linewidth=1, label='Przeżyli')
    ax.hist(died[feature], bins=20, alpha=0.6, color=COLOR_DIED, 
            edgecolor='black', linewidth=1, label='Zmarli')
    
    # Linie średnich
    ax.axvline(survived[feature].mean(), color=COLOR_SURVIVED, 
               linestyle='--', linewidth=2, label=f'Średnia (przeżyli): {survived[feature].mean():.1f}')
    ax.axvline(died[feature].mean(), color=COLOR_DIED, 
               linestyle='--', linewidth=2, label=f'Średnia (zmarli): {died[feature].mean():.1f}')
    
    ax.set_xlabel(feature_names_pl[feature], fontsize=11, fontweight='bold')
    ax.set_ylabel('Częstość', fontsize=11, fontweight='bold')
    ax.set_title(f'Rozkład: {feature_names_pl[feature]}', fontsize=12, fontweight='bold')
    ax.legend(loc='best', frameon=True, fancybox=True, shadow=True, fontsize=9)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

fig.suptitle('Rozkłady kluczowych cech klinicznych z podziałem na grupy', 
             fontsize=16, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig('../results/thesis_fig_04_distributions.png')
print("   ✓ Zapisano: thesis_fig_04_distributions.png")
plt.close()

# ============================================================================
# WYKRES 5: Macierz korelacji (heatmapa) - cechy kluczowe
# ============================================================================

print("[5/10] Generowanie: Macierz korelacji...")

fig, ax = plt.subplots(figsize=(12, 10))

# Wybór kluczowych cech + DEATH_EVENT
selected_features = key_features + ['DEATH_EVENT']
corr_matrix = df[selected_features].corr()

# Polskie nazwy
labels_pl = [feature_names_pl.get(feat, feat) for feat in selected_features[:-1]] + ['DEATH_EVENT']

# Heatmapa
sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm', 
            center=0, square=True, linewidths=2, cbar_kws={"shrink": 0.8},
            xticklabels=labels_pl, yticklabels=labels_pl,
            ax=ax, vmin=-1, vmax=1,
            annot_kws={'fontsize': 11, 'fontweight': 'bold'})

ax.set_title('Macierz korelacji kluczowych cech klinicznych', 
             fontsize=15, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('../results/thesis_fig_05_correlation_heatmap.png')
print("   ✓ Zapisano: thesis_fig_05_correlation_heatmap.png")
plt.close()

# ============================================================================
# WYKRES 6: Statystyki opisowe - tabela wizualna
# ============================================================================

print("[6/10] Generowanie: Tabela statystyk opisowych...")

fig, ax = plt.subplots(figsize=(14, 8))
ax.axis('tight')
ax.axis('off')

# Statystyki dla kluczowych cech
stats_data = []
for feature in key_features:
    row = [
        feature_names_pl[feature],
        f"{df[feature].mean():.2f}",
        f"{df[feature].median():.2f}",
        f"{df[feature].std():.2f}",
        f"{df[feature].min():.2f}",
        f"{df[feature].max():.2f}",
        f"{survived[feature].mean():.2f}",
        f"{died[feature].mean():.2f}"
    ]
    stats_data.append(row)

columns = ['Cecha', 'Średnia', 'Mediana', 'Odch. std.', 'Min', 'Max', 
           'Średnia\n(przeżyli)', 'Średnia\n(zmarli)']

table = ax.table(cellText=stats_data, colLabels=columns, 
                cellLoc='center', loc='center',
                colWidths=[0.20, 0.10, 0.10, 0.10, 0.10, 0.10, 0.15, 0.15])

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2.5)

# Stylizacja nagłówków
for i in range(len(columns)):
    cell = table[(0, i)]
    cell.set_facecolor('#3498db')
    cell.set_text_props(weight='bold', color='white', fontsize=11)

# Stylizacja wierszy
for i in range(1, len(stats_data) + 1):
    for j in range(len(columns)):
        cell = table[(i, j)]
        if i % 2 == 0:
            cell.set_facecolor('#ecf0f1')
        else:
            cell.set_facecolor('white')
        cell.set_edgecolor('black')
        cell.set_linewidth(1)

ax.set_title('Statystyki opisowe kluczowych cech klinicznych', 
             fontsize=15, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('../results/thesis_fig_06_statistics_table.png')
print("   ✓ Zapisano: thesis_fig_06_statistics_table.png")
plt.close()

# ============================================================================
# WYKRES 7: Cechy binarne - rozkład z kontekstem DEATH_EVENT
# ============================================================================

print("[7/10] Generowanie: Analiza cech binarnych...")

binary_features = ['anaemia', 'high_blood_pressure', 'diabetes', 'smoking', 'sex']
binary_labels_pl = {
    'anaemia': 'Anemia',
    'high_blood_pressure': 'Nadciśnienie',
    'diabetes': 'Cukrzyca',
    'smoking': 'Palenie',
    'sex': 'Płeć (M)'
}

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.ravel()

for idx, feature in enumerate(binary_features):
    ax = axes[idx]
    
    # Tabela kontyngencji
    contingency = pd.crosstab(df[feature], df['DEATH_EVENT'])
    
    # Wykres słupkowy zgrupowany
    x = np.arange(2)
    width = 0.35
    
    bars1 = ax.bar(x - width/2, contingency[0], width, label='Przeżyli', 
                   color=COLOR_SURVIVED, alpha=0.8, edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x + width/2, contingency[1], width, label='Zmarli', 
                   color=COLOR_DIED, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Dodanie wartości na słupkach
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{int(height)}',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_ylabel('Liczba pacjentów', fontsize=11, fontweight='bold')
    ax.set_title(binary_labels_pl[feature], fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['Nie', 'Tak'])
    ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Test chi-kwadrat
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
    significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
    
    ax.text(0.5, 0.95, f'χ²={chi2:.2f}, p={p_value:.4f} {significance}',
           transform=ax.transAxes, ha='center', va='top', fontsize=9,
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# Usunięcie pustego subplotu
fig.delaxes(axes[5])

fig.suptitle('Analiza cech binarnych w kontekście DEATH_EVENT', 
             fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('../results/thesis_fig_07_binary_features.png')
print("   ✓ Zapisano: thesis_fig_07_binary_features.png")
plt.close()

# ============================================================================
# WYKRES 8: Wartości odstające (outliers) - boxplot wszystkich cech
# ============================================================================

print("[8/10] Generowanie: Analiza wartości odstających...")

numerical_features = ['age', 'ejection_fraction', 'serum_creatinine', 
                      'serum_sodium', 'platelets', 'creatinine_phosphokinase']

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.ravel()

for idx, feature in enumerate(numerical_features):
    ax = axes[idx]
    
    # Boxplot
    bp = ax.boxplot([df[feature]], vert=True, patch_artist=True, widths=0.5,
                    boxprops=dict(facecolor=COLOR_PRIMARY, alpha=0.7, linewidth=1.5),
                    whiskerprops=dict(linewidth=1.5),
                    capprops=dict(linewidth=1.5),
                    medianprops=dict(linewidth=2, color='darkred'),
                    flierprops=dict(marker='o', markerfacecolor='red', markersize=6, 
                                   linestyle='none', markeredgecolor='darkred'))
    
    # Statystyki outlierów
    Q1 = df[feature].quantile(0.25)
    Q3 = df[feature].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[feature] < lower_bound) | (df[feature] > upper_bound)]
    
    ax.set_ylabel(feature_names_pl[feature], fontsize=11, fontweight='bold')
    ax.set_title(feature_names_pl[feature], fontsize=12, fontweight='bold')
    ax.set_xticks([])
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Adnotacja z liczbą outlierów
    textstr = f'Wartości odstające: {len(outliers)}\n({len(outliers)/len(df)*100:.1f}%)'
    textstr += f'\nZakres IQR:\n[{lower_bound:.1f}, {upper_bound:.1f}]'
    
    ax.text(0.98, 0.98, textstr, transform=ax.transAxes, 
            fontsize=9, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

fig.suptitle('Analiza wartości odstających (metoda IQR)', 
             fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('../results/thesis_fig_08_outliers.png')
print("   ✓ Zapisano: thesis_fig_08_outliers.png")
plt.close()

# ============================================================================
# WYKRES 9: Scatter plot - Age vs Ejection Fraction z kolorami DEATH_EVENT
# ============================================================================

print("[9/10] Generowanie: Scatter plot Age vs Ejection Fraction...")

fig, ax = plt.subplots(figsize=(12, 8))

# Scatter plot
scatter_survived = ax.scatter(survived['age'], survived['ejection_fraction'], 
                             c=COLOR_SURVIVED, s=80, alpha=0.6, 
                             edgecolors='black', linewidth=0.5, label='Przeżyli')
scatter_died = ax.scatter(died['age'], died['ejection_fraction'], 
                         c=COLOR_DIED, s=80, alpha=0.6, 
                         edgecolors='black', linewidth=0.5, label='Zmarli')

ax.set_xlabel('Wiek [lata]', fontsize=13, fontweight='bold')
ax.set_ylabel('Frakcja wyrzutowa [%]', fontsize=13, fontweight='bold')
ax.set_title('Zależność między wiekiem a frakcją wyrzutową\nz uwzględnieniem DEATH_EVENT', 
             fontsize=15, fontweight='bold', pad=20)
ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True, fontsize=11)
ax.grid(True, alpha=0.3, linestyle='--')

# Linie progowe kliniczne
ax.axhline(y=30, color='red', linestyle='--', linewidth=2, alpha=0.7, 
           label='Próg ciężkiej dysfunkcji (30%)')
ax.axhline(y=50, color='green', linestyle='--', linewidth=2, alpha=0.7, 
           label='Dolna granica normy (50%)')

ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True, fontsize=10)

plt.tight_layout()
plt.savefig('../results/thesis_fig_09_age_vs_ef.png')
print("   ✓ Zapisano: thesis_fig_09_age_vs_ef.png")
plt.close()

# ============================================================================
# WYKRES 10: Scatter plot - Serum Creatinine vs Serum Sodium
# ============================================================================

print("[10/10] Generowanie: Scatter plot Creatinine vs Sodium...")

fig, ax = plt.subplots(figsize=(12, 8))

# Scatter plot
scatter_survived = ax.scatter(survived['serum_creatinine'], survived['serum_sodium'], 
                             c=COLOR_SURVIVED, s=80, alpha=0.6, 
                             edgecolors='black', linewidth=0.5, label='Przeżyli')
scatter_died = ax.scatter(died['serum_creatinine'], died['serum_sodium'], 
                         c=COLOR_DIED, s=80, alpha=0.6, 
                         edgecolors='black', linewidth=0.5, label='Zmarli')

ax.set_xlabel('Kreatynina w surowicy [mg/dL]', fontsize=13, fontweight='bold')
ax.set_ylabel('Sód w surowicy [mEq/L]', fontsize=13, fontweight='bold')
ax.set_title('Zależność między kreatyniną a sodem w surowicy\nz uwzględnieniem DEATH_EVENT', 
             fontsize=15, fontweight='bold', pad=20)
ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True, fontsize=11)
ax.grid(True, alpha=0.3, linestyle='--')

# Linie progowe kliniczne
ax.axvline(x=1.2, color='red', linestyle='--', linewidth=2, alpha=0.7, 
           label='Górna granica normy kreatyniny (1.2 mg/dL)')
ax.axhline(y=135, color='orange', linestyle='--', linewidth=2, alpha=0.7, 
           label='Dolna granica normy sodu (135 mEq/L)')

ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True, fontsize=9)

plt.tight_layout()
plt.savefig('../results/thesis_fig_10_creatinine_vs_sodium.png')
print("   ✓ Zapisano: thesis_fig_10_creatinine_vs_sodium.png")
plt.close()

print("\n" + "="*80)
print("WSZYSTKIE WYKRESY ZOSTAŁY WYGENEROWANE POMYŚLNIE")
print("="*80)
print("\nWygenerowano 10 profesjonalnych wykresów do pracy inżynierskiej:")
print("  1. thesis_fig_01_target_distribution.png")
print("  2. thesis_fig_02_correlations.png")
print("  3. thesis_fig_03_key_features_comparison.png")
print("  4. thesis_fig_04_distributions.png")
print("  5. thesis_fig_05_correlation_heatmap.png")
print("  6. thesis_fig_06_statistics_table.png")
print("  7. thesis_fig_07_binary_features.png")
print("  8. thesis_fig_08_outliers.png")
print("  9. thesis_fig_09_age_vs_ef.png")
print(" 10. thesis_fig_10_creatinine_vs_sodium.png")
print("\nWszystkie wykresy zapisane w katalogu: results/")
print("Rozdzielczość: 300 DPI (odpowiednia do druku)")
print("="*80)
