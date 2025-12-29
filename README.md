# Praca Inżynierska - Badania nad Niewydolnością Serca

## 📋 Informacje o projekcie

**Tytuł:** Zastosowanie sieci neuronowych i metod uczenia maszynowego do predykcji przeżywalności pacjentów z niewydolnością serca

**Autor:** [Imię Nazwisko]

**Data rozpoczęcia:** 29 grudnia 2024

**Status:** Etap 1 - Eksploracyjna Analiza Danych (EDA) ✅ UKOŃCZONY

---

## 📊 Zbiór danych

### Opis
Zbiór danych zawiera rekordy medyczne 299 pacjentów z zaawansowaną niewydolnością serca (klasa III/IV według klasyfikacji NYHA). Wszyscy pacjenci byli zdiagnozowani z dysfunkcją skurczową lewej komory i mieli wcześniejszą historię niewydolności serca.

### Charakterystyka
- **Liczba próbek:** 299 pacjentów
- **Liczba cech:** 13 (12 cech + 1 zmienna celu)
- **Rozkład płci:** 105 kobiet, 194 mężczyzn
- **Zakres wieku:** 40-95 lat
- **Zmienna celu:** DEATH_EVENT (0 = przeżył, 1 = zmarł)

### Zmienne

| Zmienna | Typ | Opis | Zakres |
|---------|-----|------|--------|
| `age` | numeryczny | Wiek pacjenta | [40, 95] lat |
| `anaemia` | binarny | Występowanie anemii | 0/1 |
| `creatinine_phosphokinase` | numeryczny | Poziom CPK w krwi | [23, 7861] mcg/L |
| `diabetes` | binarny | Występowanie cukrzycy | 0/1 |
| `ejection_fraction` | numeryczny | Frakcja wyrzutowa serca | [14, 80] % |
| `high_blood_pressure` | binarny | Występowanie nadciśnienia | 0/1 |
| `platelets` | numeryczny | Liczba płytek krwi | [25010, 850000] /mL |
| `serum_creatinine` | numeryczny | Poziom kreatyniny w surowicy | [0.5, 9.4] mg/dL |
| `serum_sodium` | numeryczny | Poziom sodu w surowicy | [114, 148] mEq/L |
| `sex` | binarny | Płeć (0=kobieta, 1=mężczyzna) | 0/1 |
| `smoking` | binarny | Palenie papierosów | 0/1 |
| `time` | numeryczny | Okres obserwacji | [4, 285] dni |
| `DEATH_EVENT` | binarny | Zgon w okresie obserwacji | 0/1 |

---

## 🔬 Etap 1: Eksploracyjna Analiza Danych (EDA)

### Cel
Przeprowadzenie szczegółowej analizy danych w celu:
1. Zrozumienia struktury i charakterystyki danych
2. Identyfikacji wzorców i zależności między zmiennymi
3. Wykrycia potencjalnych problemów (wartości odstające, braki danych)
4. Przygotowania podstaw do dalszego modelowania

### Wykonane analizy

#### 1. Podstawowa inspekcja danych
- ✅ Wczytanie i weryfikacja struktury danych
- ✅ Sprawdzenie typów danych i brakujących wartości
- ✅ Obliczenie statystyk opisowych

**Wynik:** Dane są kompletne - brak brakujących wartości.

#### 2. Analiza zmiennej celu (DEATH_EVENT)
- ✅ Rozkład zgonów vs przeżyć
- ✅ Wizualizacja proporcji

**Wyniki:**
- **Przeżyli:** 203 pacjentów (67.89%)
- **Zmarli:** 96 pacjentów (32.11%)
- **Wniosek:** Niezbalansowanie klas - wymaga uwzględnienia w modelowaniu

#### 3. Analiza cech numerycznych
- ✅ Rozkłady wszystkich cech numerycznych
- ✅ Statystyki opisowe (średnia, mediana, odchylenie standardowe)
- ✅ Identyfikacja wartości odstających

**Kluczowe obserwacje:**
- `age`: Rozkład zbliżony do normalnego, średnia ~60 lat
- `ejection_fraction`: Średnia ~38%, wiele przypadków poniżej 30% (ciężka dysfunkcja)
- `serum_creatinine`: Rozkład prawostronnie skośny, obecność wartości odstających
- `serum_sodium`: Większość w normie (135-145 mEq/L)

#### 4. Analiza cech binarnych
- ✅ Rozkład wszystkich cech binarnych
- ✅ Proporcje dla każdej cechy

**Wyniki:**
- `sex`: 65% mężczyzn, 35% kobiet
- `smoking`: 32% palaczy
- `diabetes`: 42% z cukrzycą
- `high_blood_pressure`: 35% z nadciśnieniem
- `anaemia`: 43% z anemią

#### 5. Analiza korelacji
- ✅ Macierz korelacji wszystkich zmiennych
- ✅ Korelacje z DEATH_EVENT

**Najważniejsze korelacje z DEATH_EVENT:**
1. **time**: 0.53 (⚠️ TARGET LEAKAGE - do wykluczenia!)
2. **ejection_fraction**: -0.27 (niższa frakcja = wyższe ryzyko)
3. **serum_creatinine**: 0.29 (wyższy poziom = wyższe ryzyko)
4. **age**: 0.25 (starszy wiek = wyższe ryzyko)
5. **serum_sodium**: -0.19 (niższy poziom = wyższe ryzyko)

#### 6. Porównanie grup: Przeżyli vs Zmarli
- ✅ Wykresy pudełkowe dla wszystkich cech numerycznych
- ✅ Testy statystyczne (t-test)

**Istotne statystycznie różnice (p < 0.05):**
- `ejection_fraction`: Zmarli mieli średnio 33.5% vs 40.3% u żyjących
- `serum_creatinine`: Zmarli mieli średnio 1.84 mg/dL vs 1.19 mg/dL u żyjących
- `age`: Zmarli byli średnio starsi (65.2 lat vs 58.8 lat)
- `serum_sodium`: Zmarli mieli średnio 136.0 mEq/L vs 137.8 mEq/L u żyjących

#### 7. Analiza wartości odstających
- ✅ Wykrycie outlierów metodą IQR

**Wyniki:**
- `serum_creatinine`: 9.70% wartości odstających
- `creatinine_phosphokinase`: 9.70% wartości odstających
- `platelets`: 7.02% wartości odstających
- **Wniosek:** Wartości odstające mogą być klinicznie istotne - nie usuwać automatycznie

#### 8. Analiza cech binarnych w kontekście DEATH_EVENT
- ✅ Tabele kontyngencji
- ✅ Testy chi-kwadrat

**Wyniki:**
- `anaemia`: Istotny związek z DEATH_EVENT (p < 0.05)
- `high_blood_pressure`: Istotny związek z DEATH_EVENT (p < 0.05)
- `sex`, `smoking`, `diabetes`: Brak istotnego związku

---

## 📈 Kluczowe wnioski z EDA

### 1. Jakość danych
✅ **Dane są wysokiej jakości:**
- Brak brakujących wartości
- Wszystkie zmienne mają sensowne zakresy
- Struktura danych jest spójna

### 2. Problem niezbalansowania klas
⚠️ **Wymagana uwaga:**
- 67.89% przeżyło vs 32.11% zmarło
- Konieczne zastosowanie technik balansowania (SMOTE, class weights)

### 3. Najważniejsze cechy predykcyjne
🎯 **Top 4 cechy (bez 'time'):**
1. **ejection_fraction** (frakcja wyrzutowa)
2. **serum_creatinine** (kreatynina w surowicy)
3. **age** (wiek)
4. **serum_sodium** (sód w surowicy)

### 4. Problem TARGET LEAKAGE
🚨 **KRYTYCZNE:**
- Cecha `time` jest silnie skorelowana z DEATH_EVENT (r=0.53)
- W rzeczywistości nie znamy czasu do zgonu przed jego wystąpieniem
- **MUSI zostać wykluczona z modeli predykcyjnych**

### 5. Różnice między grupami
📊 **Pacjenci, którzy zmarli charakteryzowali się:**
- Niższą frakcją wyrzutową (33.5% vs 40.3%)
- Wyższym poziomem kreatyniny (1.84 vs 1.19 mg/dL)
- Wyższym wiekiem (65.2 vs 58.8 lat)
- Niższym poziomem sodu (136.0 vs 137.8 mEq/L)

---

## 📁 Struktura projektu

```
heart_failure_project/
│
├── data/                          # Dane
│   └── heart_failure_data.csv     # Zbiór danych Heart Failure
│
├── notebooks/                     # Notebooki z analizami
│   └── 01_exploratory_data_analysis.py
│
├── results/                       # Wyniki analiz
│   ├── 01_death_event_distribution.png
│   ├── 02_numerical_distributions.png
│   ├── 03_binary_distributions.png
│   ├── 04_correlation_matrix.png
│   ├── 05_death_event_correlations.png
│   ├── 06_survived_vs_died_comparison.png
│   ├── 07_binary_vs_death_event.png
│   └── eda_output.txt             # Pełny output z analizy
│
├── docs/                          # Dokumentacja
│   └── EDA_REPORT.md              # Szczegółowy raport z EDA
│
└── README.md                      # Ten plik
```

---

## 🚀 Następne kroki

### Etap 2: Reprodukcja analizy przeżycia
- [ ] Implementacja estymatorów Kaplana-Meiera
- [ ] Model regresji proporcjonalnych hazardów Coksa
- [ ] Wizualizacja krzywych przeżycia
- [ ] Identyfikacja czynników ryzyka

### Etap 3: Reprodukcja modeli ML bazowych
- [ ] Implementacja modeli: SVM, Random Forest, XGBoost, LightGBM
- [ ] Walidacja krzyżowa
- [ ] Porównanie z wynikami z publikacji

### Etap 4: Inżynieria cech
- [ ] Dyskretyzacja cech ciągłych
- [ ] Tworzenie cech interakcyjnych
- [ ] Normalizacja/standaryzacja

### Etap 5: Sieci neuronowe
- [ ] Implementacja MLP
- [ ] Implementacja DeepSurv
- [ ] Optymalizacja hiperparametrów
- [ ] Porównanie z modelami bazowymi

---

## 📚 Bibliografia

[1] Mishra, S. (2022). "A Comparative Study for Time-to-Event Analysis and Survival Prediction for Heart Failure Condition using Machine Learning Techniques", *Journal of Electronics, Electromedical Engineering, and Medical Informatics*, 4(3), pp. 115-134.

[2] Ahmad, T., Munir, A., Bhatti, S. H., Aftab, M., & Raza, M. A. (2017). "Survival analysis of heart failure patients: A case study", *PLOS ONE*, 12(7), e0181001.

---

## 📧 Kontakt

W razie pytań lub uwag, proszę o kontakt przez GitHub Issues.

---

**Ostatnia aktualizacja:** 29 grudnia 2024
