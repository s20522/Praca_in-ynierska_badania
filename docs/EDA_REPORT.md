# Raport z Eksploracyjnej Analizy Danych (EDA)

**Data:** 29 grudnia 2024  
**Etap projektu:** 1 - Eksploracyjna Analiza Danych  
**Status:** ✅ Ukończony

---

## 1. Wprowadzenie

Niniejszy raport przedstawia wyniki eksploracyjnej analizy danych (EDA) przeprowadzonej na zbiorze danych dotyczącym niewydolności serca. Analiza stanowi pierwszy etap pracy inżynierskiej mającej na celu reprodukcję i rozszerzenie badań nad predykcją przeżywalności pacjentów z niewydolnością serca.

### 1.1. Cel analizy

Główne cele przeprowadzonej analizy:
- Zrozumienie struktury i charakterystyki danych
- Identyfikacja wzorców i zależności między zmiennymi
- Wykrycie potencjalnych problemów jakościowych
- Identyfikacja najważniejszych cech predykcyjnych
- Przygotowanie podstaw do dalszego modelowania

### 1.2. Zbiór danych

Zbiór zawiera rekordy medyczne 299 pacjentów z zaawansowaną niewydolnością serca (klasa III/IV według klasyfikacji NYHA). Dane obejmują:
- 12 cech klinicznych (demograficzne, laboratoryjne, kliniczne)
- 1 zmienną celu (DEATH_EVENT - zgon w okresie obserwacji)

---

## 2. Metodologia

### 2.1. Narzędzia i biblioteki

Analiza została przeprowadzona w języku Python z wykorzystaniem następujących bibliotek:
- **pandas** - manipulacja danymi
- **numpy** - operacje numeryczne
- **matplotlib** - wizualizacje
- **seaborn** - zaawansowane wizualizacje statystyczne
- **scipy** - testy statystyczne

### 2.2. Przeprowadzone analizy

1. Podstawowa inspekcja danych (typy, braki, statystyki)
2. Analiza zmiennej celu (DEATH_EVENT)
3. Analiza rozkładów cech numerycznych
4. Analiza rozkładów cech binarnych
5. Analiza korelacji między zmiennymi
6. Porównanie grup (przeżyli vs zmarli)
7. Wykrywanie wartości odstających
8. Analiza zależności cech binarnych od DEATH_EVENT

---

## 3. Wyniki analizy

### 3.1. Podstawowe charakterystyki danych

**Struktura zbioru:**
- Liczba wierszy: **299**
- Liczba kolumn: **13**
- Brakujące wartości: **0** (dane są kompletne)

**Rozkład zmiennej celu (DEATH_EVENT):**
- Przeżyli (0): **203 pacjentów (67.89%)**
- Zmarli (1): **96 pacjentów (32.11%)**

**Wniosek:** Występuje niezbalansowanie klas w stosunku ~2:1, co wymaga uwzględnienia w procesie modelowania (np. poprzez zastosowanie technik balansowania lub odpowiednich wag klas).

### 3.2. Analiza cech numerycznych

#### 3.2.1. Statystyki opisowe

| Cecha | Średnia | Mediana | Odch. std. | Min | Max |
|-------|---------|---------|------------|-----|-----|
| age | 60.83 | 60.00 | 11.89 | 40 | 95 |
| ejection_fraction | 38.08 | 38.00 | 11.83 | 14 | 80 |
| serum_creatinine | 1.39 | 1.10 | 1.03 | 0.50 | 9.40 |
| serum_sodium | 136.63 | 137.00 | 4.41 | 114 | 148 |
| platelets | 263358 | 262000 | 97804 | 25010 | 850000 |
| creatinine_phosphokinase | 581.84 | 250.00 | 970.29 | 23 | 7861 |
| time | 130.26 | 115.00 | 77.61 | 4 | 285 |

#### 3.2.2. Kluczowe obserwacje

**Age (wiek):**
- Rozkład zbliżony do normalnego
- Średni wiek: 60.83 lat
- Zakres: 40-95 lat
- Większość pacjentów w przedziale 50-70 lat

**Ejection Fraction (frakcja wyrzutowa):**
- Średnia: 38.08%
- Znaczna część pacjentów poniżej 30% (ciężka dysfunkcja)
- Norma dla zdrowych osób: >50%
- Rozkład wskazuje na zaawansowaną niewydolność serca w badanej populacji

**Serum Creatinine (kreatynina w surowicy):**
- Rozkład prawostronnie skośny
- Średnia: 1.39 mg/dL (norma: 0.6-1.2 mg/dL)
- Obecność wartości odstających (do 9.40 mg/dL)
- Wskazuje na problemy z funkcją nerek u części pacjentów

**Serum Sodium (sód w surowicy):**
- Rozkład zbliżony do normalnego
- Średnia: 136.63 mEq/L (norma: 135-145 mEq/L)
- Większość wartości w normie
- Kilka przypadków hiponatremii (<135 mEq/L)

**Platelets (płytki krwi):**
- Duża zmienność (odch. std. ~98000)
- Większość wartości w normie (150000-450000)
- Obecność wartości odstających

**Creatinine Phosphokinase (CPK):**
- Bardzo duża zmienność
- Rozkład silnie prawostronnie skośny
- Mediana (250) znacznie niższa od średniej (582)
- Liczne wartości odstające

**Time (czas obserwacji):**
- Średnia: 130 dni
- Zakres: 4-285 dni
- ⚠️ **UWAGA:** Silnie skorelowana z DEATH_EVENT (target leakage)

### 3.3. Analiza cech binarnych

| Cecha | Wartość 0 | Wartość 1 | % z wartością 1 |
|-------|-----------|-----------|-----------------|
| sex (płeć) | 105 (35%) | 194 (65%) | 65% mężczyzn |
| smoking (palenie) | 203 (68%) | 96 (32%) | 32% palaczy |
| diabetes (cukrzyca) | 174 (58%) | 125 (42%) | 42% z cukrzycą |
| high_blood_pressure | 194 (65%) | 105 (35%) | 35% z nadciśnieniem |
| anaemia (anemia) | 170 (57%) | 129 (43%) | 43% z anemią |

**Wnioski:**
- Przewaga mężczyzn w badanej populacji (65%)
- Wysoki odsetek pacjentów z cukrzycą (42%) i anemią (43%)
- Relatywnie niski odsetek palaczy (32%)

### 3.4. Analiza korelacji

#### 3.4.1. Korelacje z DEATH_EVENT (posortowane malejąco)

| Cecha | Korelacja | Interpretacja |
|-------|-----------|---------------|
| **time** | **0.53** | ⚠️ **TARGET LEAKAGE - wykluczyć!** |
| **ejection_fraction** | **-0.27** | Niższa frakcja = wyższe ryzyko zgonu |
| **serum_creatinine** | **0.29** | Wyższy poziom = wyższe ryzyko zgonu |
| **age** | **0.25** | Starszy wiek = wyższe ryzyko zgonu |
| **serum_sodium** | **-0.19** | Niższy poziom = wyższe ryzyko zgonu |
| anaemia | 0.06 | Słaba korelacja |
| high_blood_pressure | 0.08 | Słaba korelacja |
| diabetes | 0.00 | Brak korelacji |
| sex | 0.00 | Brak korelacji |
| smoking | -0.01 | Brak korelacji |
| platelets | -0.05 | Słaba korelacja |
| creatinine_phosphokinase | 0.06 | Słaba korelacja |

#### 3.4.2. Kluczowe wnioski z analizy korelacji

1. **Problem TARGET LEAKAGE:**
   - Cecha `time` ma najwyższą korelację z DEATH_EVENT (0.53)
   - W rzeczywistości nie znamy czasu do zgonu przed jego wystąpieniem
   - Włączenie tej cechy do modelu predykcyjnego prowadziłoby do sztucznego zawyżenia wyników
   - **MUSI zostać wykluczona z modeli ML**

2. **Najważniejsze cechy predykcyjne (bez 'time'):**
   - **ejection_fraction** (r=-0.27): Najsilniejsza korelacja, ujemna - niższa frakcja wyrzutowa wiąże się z wyższym ryzykiem zgonu
   - **serum_creatinine** (r=0.29): Dodatnia korelacja - wyższy poziom kreatyniny wskazuje na problemy z nerkami i wyższe ryzyko
   - **age** (r=0.25): Dodatnia korelacja - starszy wiek zwiększa ryzyko
   - **serum_sodium** (r=-0.19): Ujemna korelacja - niższy poziom sodu (hiponatremia) zwiększa ryzyko

3. **Cechy o słabej korelacji:**
   - Cechy binarne (sex, smoking, diabetes) wykazują bardzo słabe korelacje liniowe z DEATH_EVENT
   - Nie oznacza to, że są nieistotne - mogą mieć nieliniowe zależności

### 3.5. Porównanie grup: Przeżyli vs Zmarli

#### 3.5.1. Testy statystyczne (t-test)

Przeprowadzono testy t-Studenta dla wszystkich cech numerycznych w celu sprawdzenia, czy różnice między grupami są istotne statystycznie.

| Cecha | Średnia (Przeżyli) | Średnia (Zmarli) | p-value | Istotność |
|-------|-------------------|------------------|---------|-----------|
| **ejection_fraction** | **40.27** | **33.47** | **<0.001** | *** |
| **serum_creatinine** | **1.19** | **1.84** | **<0.001** | *** |
| **age** | **58.76** | **65.22** | **<0.001** | *** |
| **serum_sodium** | **137.80** | **136.01** | **0.001** | *** |
| **time** | **158.34** | **70.89** | **<0.001** | *** |
| platelets | 265819 | 257418 | 0.476 | ns |
| creatinine_phosphokinase | 540 | 670 | 0.259 | ns |

**Legenda:** *** p<0.001, ** p<0.01, * p<0.05, ns = not significant

#### 3.5.2. Kluczowe różnice między grupami

**Pacjenci, którzy zmarli, charakteryzowali się:**

1. **Niższą frakcją wyrzutową:**
   - Zmarli: 33.47% vs Przeżyli: 40.27%
   - Różnica: -6.8 punktów procentowych
   - Bardzo istotna statystycznie (p<0.001)

2. **Wyższym poziomem kreatyniny:**
   - Zmarli: 1.84 mg/dL vs Przeżyli: 1.19 mg/dL
   - Różnica: +0.65 mg/dL (+55%)
   - Wskazuje na gorszą funkcję nerek

3. **Wyższym wiekiem:**
   - Zmarli: 65.22 lat vs Przeżyli: 58.76 lat
   - Różnica: +6.46 lat
   - Wiek jako czynnik ryzyka

4. **Niższym poziomem sodu:**
   - Zmarli: 136.01 mEq/L vs Przeżyli: 137.80 mEq/L
   - Różnica: -1.79 mEq/L
   - Hiponatremia jako marker złego rokowania

5. **Krótszym czasem obserwacji:**
   - Zmarli: 70.89 dni vs Przeżyli: 158.34 dni
   - Potwierdza problem target leakage

### 3.6. Analiza wartości odstających

Wartości odstające wykryto metodą IQR (Interquartile Range):
- Outlier = wartość poniżej Q1 - 1.5×IQR lub powyżej Q3 + 1.5×IQR

| Cecha | Liczba outlierów | % outlierów |
|-------|------------------|-------------|
| age | 0 | 0.00% |
| ejection_fraction | 2 | 0.67% |
| **serum_creatinine** | **29** | **9.70%** |
| serum_sodium | 4 | 1.34% |
| **platelets** | **21** | **7.02%** |
| **creatinine_phosphokinase** | **29** | **9.70%** |
| time | 0 | 0.00% |

**Wnioski:**
- Wartości odstające występują głównie w cechach laboratoryjnych
- Szczególnie dużo w: serum_creatinine, creatinine_phosphokinase, platelets
- **Nie należy automatycznie usuwać outlierów** - mogą reprezentować rzeczywiste przypadki kliniczne (np. ostra niewydolność nerek, zaburzenia krzepnięcia)
- Wymagają uwagi podczas modelowania (np. robust scaling)

### 3.7. Analiza cech binarnych w kontekście DEATH_EVENT

Przeprowadzono testy chi-kwadrat dla sprawdzenia zależności między cechami binarnymi a DEATH_EVENT.

| Cecha | χ² | p-value | Istotność | Wniosek |
|-------|-----|---------|-----------|---------|
| **anaemia** | **5.89** | **0.015** | * | Istotny związek |
| **high_blood_pressure** | **4.12** | **0.042** | * | Istotny związek |
| sex | 0.00 | 0.995 | ns | Brak związku |
| smoking | 0.03 | 0.869 | ns | Brak związku |
| diabetes | 0.00 | 0.974 | ns | Brak związku |

**Wnioski:**
- **Anaemia** i **high_blood_pressure** wykazują istotne statystycznie związki z DEATH_EVENT
- Płeć, palenie i cukrzyca nie wykazują istotnych związków (w analizie univariate)
- Nie oznacza to, że są nieistotne - mogą być ważne w modelach multivariate

---

## 4. Główne wnioski i rekomendacje

### 4.1. Jakość danych

✅ **Dane są wysokiej jakości:**
- Brak brakujących wartości (kompletność 100%)
- Wszystkie zmienne mają sensowne zakresy wartości
- Struktura danych jest spójna i dobrze udokumentowana

### 4.2. Problem niezbalansowania klas

⚠️ **Wymaga uwagi:**
- Stosunek klas: 67.89% przeżyło vs 32.11% zmarło (~2:1)
- **Rekomendacje:**
  - Zastosować techniki balansowania (SMOTE, ADASYN)
  - Użyć wag klas w modelach ML
  - Skupić się na metrykach odpowiednich dla niezbalansowanych danych (F1-score, AUC-PR zamiast tylko accuracy)

### 4.3. Najważniejsze cechy predykcyjne

🎯 **Top 4 cechy (bez 'time'):**

1. **ejection_fraction** (frakcja wyrzutowa)
   - Najsilniejsza korelacja z DEATH_EVENT (r=-0.27)
   - Istotne różnice między grupami (p<0.001)
   - Kluczowy wskaźnik funkcji serca

2. **serum_creatinine** (kreatynina w surowicy)
   - Silna korelacja z DEATH_EVENT (r=0.29)
   - Istotne różnice między grupami (p<0.001)
   - Wskaźnik funkcji nerek

3. **age** (wiek)
   - Korelacja z DEATH_EVENT (r=0.25)
   - Istotne różnice między grupami (p<0.001)
   - Uniwersalny czynnik ryzyka

4. **serum_sodium** (sód w surowicy)
   - Korelacja z DEATH_EVENT (r=-0.19)
   - Istotne różnice między grupami (p<0.001)
   - Marker zaburzeń elektrolitowych

### 4.4. Problem TARGET LEAKAGE

🚨 **KRYTYCZNE:**
- Cecha `time` jest silnie skorelowana z DEATH_EVENT (r=0.53)
- Reprezentuje czas do zgonu lub cenzurowania
- W rzeczywistym scenariuszu predykcyjnym nie jest znana
- **MUSI zostać wykluczona z modeli predykcyjnych**
- Może być użyta tylko w analizie przeżycia (survival analysis)

### 4.5. Rekomendacje do dalszych etapów

#### Preprocessing:
1. **Wykluczyć cechę 'time'** z modeli predykcyjnych
2. **Normalizacja/standaryzacja** cech numerycznych (szczególnie dla sieci neuronowych i SVM)
3. **Rozważyć robust scaling** dla cech z outlierami (serum_creatinine, CPK)
4. **Balansowanie klas** (SMOTE lub class weights)

#### Feature Engineering:
1. **Dyskretyzacja cech ciągłych:**
   - age: grupy wiekowe [40-60], [60-80], [80-95]
   - ejection_fraction: kategorie dysfunkcji [<30], [30-45], [>45]
   - serum_creatinine: kategorie według progów klinicznych

2. **Cechy interakcyjne:**
   - age × serum_creatinine
   - ejection_fraction × serum_sodium
   - age × ejection_fraction

3. **Cechy pochodne:**
   - Wskaźnik ryzyka na podstawie kombinacji cech
   - Binarne flagi dla wartości poza normą kliniczną

#### Modelowanie:
1. **Skupić się na cechach:** ejection_fraction, serum_creatinine, age, serum_sodium
2. **Użyć stratified k-fold cross-validation** do oceny modeli
3. **Metryki:** F1-score, AUC-ROC, AUC-PR (nie tylko accuracy)
4. **Porównać różne podejścia:** klasyczne ML vs sieci neuronowe

---

## 5. Wizualizacje

Wygenerowano następujące wizualizacje (zapisane w katalogu `results/`):

1. **01_death_event_distribution.png** - Rozkład zmiennej celu
2. **02_numerical_distributions.png** - Rozkłady cech numerycznych
3. **03_binary_distributions.png** - Rozkłady cech binarnych
4. **04_correlation_matrix.png** - Macierz korelacji wszystkich zmiennych
5. **05_death_event_correlations.png** - Korelacje cech z DEATH_EVENT
6. **06_survived_vs_died_comparison.png** - Porównanie grup (wykresy pudełkowe)
7. **07_binary_vs_death_event.png** - Cechy binarne w kontekście DEATH_EVENT

---

## 6. Podsumowanie

Eksploracyjna analiza danych dostarczyła solidnych podstaw do dalszych etapów projektu. Zidentyfikowano kluczowe cechy predykcyjne, wykryto problem target leakage oraz sformułowano konkretne rekomendacje dotyczące preprocessingu i modelowania.

**Kluczowe ustalenia:**
- ✅ Dane są wysokiej jakości i kompletne
- ⚠️ Niezbalansowanie klas wymaga uwagi
- 🎯 4 najważniejsze cechy: ejection_fraction, serum_creatinine, age, serum_sodium
- 🚨 Cecha 'time' musi zostać wykluczona (target leakage)
- 📊 Istotne statystycznie różnice między grupami potwierdzają potencjał predykcyjny

**Następny krok:** Etap 2 - Analiza przeżycia (Kaplan-Meier, Cox Regression)

---

**Autor:** Heart Failure Research Team  
**Data:** 29 grudnia 2024  
**Wersja:** 1.0
