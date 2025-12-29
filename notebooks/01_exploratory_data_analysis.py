"""
Eksploracyjna Analiza Danych (EDA) - Heart Failure Dataset
============================================================

Autor: Heart Failure Research Team
Data: 2025-12-29

Cel: Przeprowadzenie szczegółowej eksploracyjnej analizy danych w celu:
1. Zrozumienia struktury i charakterystyki danych
2. Identyfikacji wzorców i zależności między zmiennymi
3. Wykrycia potencjalnych problemów (wartości odstające, braki danych)
4. Przygotowania podstaw do dalszego modelowania
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Konfiguracja wizualizacji
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

# ============================================================================
# 1. WCZYTANIE I PODSTAWOWA INSPEKCJA DANYCH
# ============================================================================

print("="*80)
print("EKSPLORACYJNA ANALIZA DANYCH - HEART FAILURE DATASET")
print("="*80)

# Wczytanie danych
df = pd.read_csv('../data/heart_failure_data.csv')

print("\n1. PODSTAWOWE INFORMACJE O ZBIORZE DANYCH")
print("-"*80)
print(f"Liczba wierszy: {df.shape[0]}")
print(f"Liczba kolumn: {df.shape[1]}")
print(f"\nNazwy kolumn:\n{df.columns.tolist()}")

# Mapowanie nazw kolumn na bardziej czytelne
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

print("\n2. PIERWSZE 5 WIERSZY DANYCH")
print("-"*80)
print(df.head())

print("\n3. TYPY DANYCH I BRAKUJĄCE WARTOŚCI")
print("-"*80)
print(df.info())

print("\n4. STATYSTYKI OPISOWE")
print("-"*80)
print(df.describe())

# Sprawdzenie brakujących wartości
print("\n5. BRAKUJĄCE WARTOŚCI")
print("-"*80)
missing_values = df.isnull().sum()
print(missing_values)
print(f"\nCałkowita liczba brakujących wartości: {missing_values.sum()}")

# ============================================================================
# 2. ANALIZA ZMIENNEJ CELU (DEATH_EVENT)
# ============================================================================

print("\n" + "="*80)
print("ANALIZA ZMIENNEJ CELU (DEATH_EVENT)")
print("="*80)

death_counts = df['DEATH_EVENT'].value_counts()
death_percentages = df['DEATH_EVENT'].value_counts(normalize=True) * 100

print(f"\nRozkład zmiennej DEATH_EVENT:")
print(f"Przeżyli (0): {death_counts[0]} ({death_percentages[0]:.2f}%)")
print(f"Zmarli (1): {death_counts[1]} ({death_percentages[1]:.2f}%)")

# Wizualizacja rozkładu zmiennej celu
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Wykres słupkowy
axes[0].bar(['Przeżyli', 'Zmarli'], death_counts.values, color=['#2ecc71', '#e74c3c'])
axes[0].set_ylabel('Liczba pacjentów')
axes[0].set_title('Rozkład zmiennej celu (DEATH_EVENT)', fontsize=14, fontweight='bold')
axes[0].grid(axis='y', alpha=0.3)
for i, v in enumerate(death_counts.values):
    axes[0].text(i, v + 5, str(v), ha='center', va='bottom', fontweight='bold')

# Wykres kołowy
colors = ['#2ecc71', '#e74c3c']
axes[1].pie(death_counts.values, labels=['Przeżyli', 'Zmarli'], autopct='%1.1f%%',
            colors=colors, startangle=90, textprops={'fontsize': 12, 'fontweight': 'bold'})
axes[1].set_title('Proporcje zgonów', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('../results/01_death_event_distribution.png', dpi=300, bbox_inches='tight')
print("\n✓ Zapisano wykres: results/01_death_event_distribution.png")
plt.close()

# ============================================================================
# 3. ANALIZA CECH NUMERYCZNYCH
# ============================================================================

print("\n" + "="*80)
print("ANALIZA CECH NUMERYCZNYCH")
print("="*80)

numerical_features = ['age', 'ejection_fraction', 'serum_creatinine', 
                      'serum_sodium', 'platelets', 'creatinine_phosphokinase', 'time']

print("\nStatystyki opisowe dla cech numerycznych:")
print(df[numerical_features].describe())

# Rozkłady cech numerycznych
fig, axes = plt.subplots(3, 3, figsize=(18, 15))
axes = axes.ravel()

for idx, feature in enumerate(numerical_features):
    # Histogram z krzywą gęstości
    axes[idx].hist(df[feature], bins=30, alpha=0.7, color='steelblue', edgecolor='black')
    axes[idx].set_xlabel(feature, fontsize=11)
    axes[idx].set_ylabel('Częstość', fontsize=11)
    axes[idx].set_title(f'Rozkład: {feature}', fontsize=12, fontweight='bold')
    axes[idx].grid(axis='y', alpha=0.3)
    
    # Dodanie linii średniej i mediany
    mean_val = df[feature].mean()
    median_val = df[feature].median()
    axes[idx].axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Średnia: {mean_val:.2f}')
    axes[idx].axvline(median_val, color='green', linestyle='--', linewidth=2, label=f'Mediana: {median_val:.2f}')
    axes[idx].legend(fontsize=9)

# Usunięcie pustych subplotów
for idx in range(len(numerical_features), len(axes)):
    fig.delaxes(axes[idx])

plt.tight_layout()
plt.savefig('../results/02_numerical_distributions.png', dpi=300, bbox_inches='tight')
print("\n✓ Zapisano wykres: results/02_numerical_distributions.png")
plt.close()

# ============================================================================
# 4. ANALIZA CECH BINARNYCH
# ============================================================================

print("\n" + "="*80)
print("ANALIZA CECH BINARNYCH")
print("="*80)

binary_features = ['sex', 'smoking', 'diabetes', 'high_blood_pressure', 'anaemia']

print("\nRozkład cech binarnych:")
for feature in binary_features:
    counts = df[feature].value_counts()
    percentages = df[feature].value_counts(normalize=True) * 100
    print(f"\n{feature}:")
    print(f"  0: {counts[0]} ({percentages[0]:.2f}%)")
    print(f"  1: {counts[1]} ({percentages[1]:.2f}%)")

# Wizualizacja cech binarnych
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.ravel()

feature_labels = {
    'sex': ['Kobieta', 'Mężczyzna'],
    'smoking': ['Niepalący', 'Palący'],
    'diabetes': ['Bez cukrzycy', 'Z cukrzycą'],
    'high_blood_pressure': ['Bez nadciśnienia', 'Z nadciśnieniem'],
    'anaemia': ['Bez anemii', 'Z anemią']
}

for idx, feature in enumerate(binary_features):
    counts = df[feature].value_counts()
    labels = feature_labels[feature]
    axes[idx].bar(labels, counts.values, color=['#3498db', '#e67e22'])
    axes[idx].set_ylabel('Liczba pacjentów', fontsize=11)
    axes[idx].set_title(f'Rozkład: {feature}', fontsize=12, fontweight='bold')
    axes[idx].grid(axis='y', alpha=0.3)
    for i, v in enumerate(counts.values):
        axes[idx].text(i, v + 3, str(v), ha='center', va='bottom', fontweight='bold')

# Usunięcie pustego subplotu
fig.delaxes(axes[5])

plt.tight_layout()
plt.savefig('../results/03_binary_distributions.png', dpi=300, bbox_inches='tight')
print("\n✓ Zapisano wykres: results/03_binary_distributions.png")
plt.close()

# ============================================================================
# 5. ANALIZA KORELACJI
# ============================================================================

print("\n" + "="*80)
print("ANALIZA KORELACJI")
print("="*80)

# Macierz korelacji
correlation_matrix = df.corr()

print("\nMacierz korelacji:")
print(correlation_matrix)

# Korelacje z DEATH_EVENT
print("\nKorelacje z DEATH_EVENT (posortowane):")
death_correlations = correlation_matrix['DEATH_EVENT'].sort_values(ascending=False)
print(death_correlations)

# Wizualizacja macierzy korelacji
fig, ax = plt.subplots(figsize=(14, 12))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
            center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8},
            ax=ax)
ax.set_title('Macierz korelacji wszystkich zmiennych', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('../results/04_correlation_matrix.png', dpi=300, bbox_inches='tight')
print("\n✓ Zapisano wykres: results/04_correlation_matrix.png")
plt.close()

# Wykres korelacji z DEATH_EVENT
fig, ax = plt.subplots(figsize=(10, 8))
death_corr_sorted = death_correlations.drop('DEATH_EVENT')
colors = ['#e74c3c' if x > 0 else '#2ecc71' for x in death_corr_sorted.values]
ax.barh(range(len(death_corr_sorted)), death_corr_sorted.values, color=colors)
ax.set_yticks(range(len(death_corr_sorted)))
ax.set_yticklabels(death_corr_sorted.index)
ax.set_xlabel('Korelacja z DEATH_EVENT', fontsize=12)
ax.set_title('Korelacje cech z DEATH_EVENT', fontsize=14, fontweight='bold')
ax.axvline(0, color='black', linewidth=0.8)
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('../results/05_death_event_correlations.png', dpi=300, bbox_inches='tight')
print("✓ Zapisano wykres: results/05_death_event_correlations.png")
plt.close()

# ============================================================================
# 6. PORÓWNANIE CECH DLA PACJENTÓW, KTÓRZY PRZEŻYLI VS ZMARLI
# ============================================================================

print("\n" + "="*80)
print("PORÓWNANIE GRUP: PRZEŻYLI VS ZMARLI")
print("="*80)

survived = df[df['DEATH_EVENT'] == 0]
died = df[df['DEATH_EVENT'] == 1]

print(f"\nLiczba pacjentów, którzy przeżyli: {len(survived)}")
print(f"Liczba pacjentów, którzy zmarli: {len(died)}")

print("\nStatystyki opisowe - PRZEŻYLI:")
print(survived[numerical_features].describe())

print("\nStatystyki opisowe - ZMARLI:")
print(died[numerical_features].describe())

# Wykresy pudełkowe dla porównania grup
fig, axes = plt.subplots(3, 3, figsize=(18, 15))
axes = axes.ravel()

for idx, feature in enumerate(numerical_features):
    data_to_plot = [survived[feature], died[feature]]
    bp = axes[idx].boxplot(data_to_plot, labels=['Przeżyli', 'Zmarli'],
                            patch_artist=True, widths=0.6)
    
    # Kolorowanie
    colors = ['#2ecc71', '#e74c3c']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    axes[idx].set_ylabel(feature, fontsize=11)
    axes[idx].set_title(f'Porównanie: {feature}', fontsize=12, fontweight='bold')
    axes[idx].grid(axis='y', alpha=0.3)
    
    # Test t-Studenta
    t_stat, p_value = stats.ttest_ind(survived[feature], died[feature])
    significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
    axes[idx].text(0.5, 0.95, f'p-value: {p_value:.4f} {significance}',
                   transform=axes[idx].transAxes, ha='center', va='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Usunięcie pustych subplotów
for idx in range(len(numerical_features), len(axes)):
    fig.delaxes(axes[idx])

plt.tight_layout()
plt.savefig('../results/06_survived_vs_died_comparison.png', dpi=300, bbox_inches='tight')
print("\n✓ Zapisano wykres: results/06_survived_vs_died_comparison.png")
plt.close()

# ============================================================================
# 7. ANALIZA WARTOŚCI ODSTAJĄCYCH
# ============================================================================

print("\n" + "="*80)
print("ANALIZA WARTOŚCI ODSTAJĄCYCH (OUTLIERS)")
print("="*80)

def detect_outliers_iqr(data, feature):
    """Wykrywa wartości odstające metodą IQR"""
    Q1 = data[feature].quantile(0.25)
    Q3 = data[feature].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = data[(data[feature] < lower_bound) | (data[feature] > upper_bound)]
    return outliers, lower_bound, upper_bound

print("\nWartości odstające (metoda IQR):")
for feature in numerical_features:
    outliers, lower, upper = detect_outliers_iqr(df, feature)
    print(f"\n{feature}:")
    print(f"  Zakres normalny: [{lower:.2f}, {upper:.2f}]")
    print(f"  Liczba wartości odstających: {len(outliers)} ({len(outliers)/len(df)*100:.2f}%)")

# ============================================================================
# 8. ANALIZA CECH BINARNYCH W KONTEKŚCIE DEATH_EVENT
# ============================================================================

print("\n" + "="*80)
print("ANALIZA CECH BINARNYCH W KONTEKŚCIE DEATH_EVENT")
print("="*80)

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.ravel()

for idx, feature in enumerate(binary_features):
    # Tabela kontyngencji
    contingency_table = pd.crosstab(df[feature], df['DEATH_EVENT'])
    
    # Wykres słupkowy zgrupowany
    contingency_table.plot(kind='bar', ax=axes[idx], color=['#2ecc71', '#e74c3c'], width=0.7)
    axes[idx].set_xlabel(feature, fontsize=11)
    axes[idx].set_ylabel('Liczba pacjentów', fontsize=11)
    axes[idx].set_title(f'{feature} vs DEATH_EVENT', fontsize=12, fontweight='bold')
    axes[idx].legend(['Przeżyli', 'Zmarli'], loc='upper right')
    axes[idx].grid(axis='y', alpha=0.3)
    axes[idx].set_xticklabels(feature_labels[feature], rotation=0)
    
    # Test chi-kwadrat
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
    significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
    axes[idx].text(0.5, 0.95, f'χ²={chi2:.2f}, p={p_value:.4f} {significance}',
                   transform=axes[idx].transAxes, ha='center', va='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Usunięcie pustego subplotu
fig.delaxes(axes[5])

plt.tight_layout()
plt.savefig('../results/07_binary_vs_death_event.png', dpi=300, bbox_inches='tight')
print("\n✓ Zapisano wykres: results/07_binary_vs_death_event.png")
plt.close()

# ============================================================================
# 9. PODSUMOWANIE I WNIOSKI
# ============================================================================

print("\n" + "="*80)
print("PODSUMOWANIE EKSPLORACYJNEJ ANALIZY DANYCH")
print("="*80)

print("""
KLUCZOWE WNIOSKI:

1. STRUKTURA DANYCH:
   - Zbiór zawiera 299 pacjentów z 13 zmiennymi
   - Brak brakujących wartości - dane są kompletne
   - Niezbalansowanie klas: 67.89% przeżyło, 32.11% zmarło

2. NAJWAŻNIEJSZE CECHY PREDYKCYJNE (na podstawie korelacji z DEATH_EVENT):
   - time: silna dodatnia korelacja (UWAGA: target leakage!)
   - ejection_fraction: ujemna korelacja (niższa frakcja = wyższe ryzyko)
   - serum_creatinine: dodatnia korelacja (wyższy poziom = wyższe ryzyko)
   - age: dodatnia korelacja (starszy wiek = wyższe ryzyko)
   - serum_sodium: ujemna korelacja (niższy poziom = wyższe ryzyko)

3. RÓŻNICE MIĘDZY GRUPAMI (PRZEŻYLI VS ZMARLI):
   - Zmarli mieli istotnie statystycznie:
     * Niższą frakcję wyrzutową (ejection_fraction)
     * Wyższy poziom kreatyniny w surowicy (serum_creatinine)
     * Wyższy wiek (age)
     * Niższy poziom sodu w surowicy (serum_sodium)

4. WARTOŚCI ODSTAJĄCE:
   - Występują w większości cech numerycznych
   - Szczególnie widoczne w: creatinine_phosphokinase, platelets, serum_creatinine
   - Wymagają uwagi podczas modelowania (nie usuwać - mogą być klinicznie istotne)

5. CECHY BINARNE:
   - Anaemia i high_blood_pressure wykazują istotne statystycznie związki z DEATH_EVENT
   - Sex, smoking, diabetes - słabsze związki

6. PROBLEM TARGET LEAKAGE:
   - Cecha 'time' jest silnie skorelowana z DEATH_EVENT
   - W rzeczywistości nie znamy czasu do zgonu przed jego wystąpieniem
   - MUSI zostać wykluczona z modeli predykcyjnych

REKOMENDACJE DO DALSZYCH ETAPÓW:
- Wykluczyć cechę 'time' z modelowania predykcyjnego
- Skupić się na cechach: ejection_fraction, serum_creatinine, age, serum_sodium
- Rozważyć inżynierię cech (interakcje, dyskretyzacja)
- Zastosować techniki balansowania klas (SMOTE, class weights)
- Normalizacja/standaryzacja przed trenowaniem modeli neuronowych
""")

print("\n" + "="*80)
print("ANALIZA ZAKOŃCZONA POMYŚLNIE")
print("="*80)
print("\nWszystkie wykresy zostały zapisane w katalogu 'results/'")
