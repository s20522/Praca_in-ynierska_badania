# Szkielet Pracy Inżynierskiej: Predykcja Niewydolności Serca

**Tytuł:** *Zastosowanie modeli uczenia maszynowego, w tym sieci neuronowych, do predykcji śmiertelności pacjentów z niewydolnością serca.*

**Autor:** [Twoje Imię i Nazwisko]

**Repozytorium z kodem:** [https://github.com/s20522/Praca_in-ynierska_badania](https://github.com/s20522/Praca_in-ynierska_badania)

---

### Streszczenie

*(~150-200 słów) Krótkie podsumowanie problemu, metodologii, kluczowych wyników (Random Forest najlepszy, F1=0.68, Recall=89.5%) i wniosków (klasyczne modele > MLP na małym zbiorze).*

---

### 1. Wstęp

1.1. **Kontekst medyczny:** Problem niewydolności serca, statystyki, znaczenie wczesnej predykcji.
1.2. **Problem badawczy:** Potrzeba tworzenia dokładnych modeli predykcyjnych.
1.3. **Cel pracy:** Porównanie skuteczności klasycznych modeli uczenia maszynowego (Random Forest) z sieciami neuronowymi (MLP) w zadaniu predykcji śmiertelności.
1.4. **Struktura pracy:** Krótki opis kolejnych rozdziałów.

---

### 2. Przegląd Literatury (Literature Review)

2.1. **Uczenie maszynowe w kardiologii:** Przykłady zastosowań (np. EKG, obrazowanie).
2.2. **Modele predykcyjne w niewydolności serca:** Omówienie istniejących badań, w tym pracy bazowej [1].
2.3. **Wybrane algorytmy:**
    - **Random Forest:** Krótki opis działania, zalety (odporność, interpretowalność).
    - **Sieci Neuronowe (MLP):** Opis, potencjał w modelowaniu złożonych zależności, wyzwania (ilość danych, przeuczenie).

---

### 3. Metodologia i Środowisko Badawcze

3.1. **Zbiór danych:**
    - Opis zbioru (299 pacjentów, 13 cech), źródło [2].
    - **Tabela 1:** *Charakterystyka zbioru danych* (wstawić `results/eda_fig_06_stats_table.png`).
3.2. **Eksploracyjna Analiza Danych (EDA):**
    - Opis przeprowadzonych kroków (analiza rozkładów, korelacji, wartości odstających).
    - **Wykres 1:** *Rozkład zmiennej celu* (wstawić `results/thesis_fig_01_target_distribution.png`).
    - **Wykres 2:** *Porównanie kluczowych cech dla obu grup* (wstawić `results/thesis_fig_03_key_features_comparison.png`).
    - Identyfikacja problemu "target leakage" (cecha `time`).
3.3. **Metryki oceny:** Wyjaśnienie F1-score, Recall, Precision, AUC-ROC i ich znaczenia w kontekście medycznym.
3.4. **Środowisko pracy:** Python 3.11, biblioteki (scikit-learn, TensorFlow, pandas), GitHub.

---

### 4. Przeprowadzone Eksperymenty i Wyniki

4.1. **Model Bazowy: Random Forest**
    - Opis budowy modelu (3 kluczowe cechy, standaryzacja, walidacja krzyżowa).
    - **Wykres 3:** *Macierz pomyłek dla modelu Random Forest* (wstawić `results/rf_fig_01_confusion_matrix.png`).
    - **Tabela 2:** *Wyniki modelu Random Forest* (wstawić `results/rf_fig_05_metrics_summary.png`).
    - Omówienie wyników (F1=0.68, Recall=89.5%).

4.2. **Eksperymenty z Inżynierią Cech (Feature Engineering)**
    - Opis testowanych technik (dyskretyzacja, interakcje).
    - **Wykres 4:** *Porównanie metryk dla różnych technik inżynierii cech* (wstawić `results/fe_fig_01_metrics_comparison.png`).
    - Wniosek: Brak poprawy, model bazowy pozostał najlepszy.

4.3. **Eksperymenty z Sieciami Neuronowymi (MLP)**
    - Opis systematycznej optymalizacji (architektura, aktywacje, regularyzacja, optymalizatory).
    - **Wykres 5:** *Porównanie najlepszych modeli MLP z Random Forest* (wstawić `results/nn_fig_06_mlp_vs_rf.png`).
    - **Tabela 3:** *Top 10 modeli ze wszystkich eksperymentów* (wstawić `results/nn_fig_07_summary_table.png`).
    - Wniosek: MLP nie pokonał RF, głównie z powodu niższego Recall.

---

### 5. Dyskusja i Wnioski

5.1. **Dyskusja wyników:**
    - Analiza, dlaczego Random Forest okazał się lepszy (odporność na małe zbiory, prostota).
    - Omówienie, dlaczego MLP nie osiągnął lepszych wyników (mały zbiór danych, ryzyko przeuczenia).
    - Znaczenie wysokiego Recall w zastosowaniach medycznych.
5.2. **Wnioski końcowe:**
    - Odpowiedź na cel pracy: W badanym przypadku klasyczny model RF jest lepszym wyborem.
    - Potwierdzenie, że "więcej" (złożoności, cech) nie zawsze znaczy "lepiej".
5.3. **Ograniczenia pracy i kierunki dalszych badań:**
    - Mały zbiór danych jako główne ograniczenie.
    - Propozycje: testowanie na większym zbiorze, inne architektury (np. 1D-CNN), zaawansowane metody analizy przeżycia (DeepSurv).

---

### Bibliografia

[1] Chicco, D., & Jurman, G. (2020). Machine learning can predict survival of patients with heart failure from serum creatinine and ejection fraction alone. *BMC Medical Informatics and Decision Making*, 20(1), 16.

[2] Ahmad, T., Munir, A., Bhatti, S. H., Aftab, M., & Raza, M. A. (2017). Survival analysis of heart failure patients: a case study. *PloS one*, 12(7), e0181001.

---

### Spis Tabel i Wykresów

- **Tabela 1:** Charakterystyka zbioru danych.
- **Tabela 2:** Wyniki modelu Random Forest.
- **Tabela 3:** Top 10 modeli ze wszystkich eksperymentów.
- **Wykres 1:** Rozkład zmiennej celu.
- **Wykres 2:** Porównanie kluczowych cech dla obu grup.
- **Wykres 3:** Macierz pomyłek dla modelu Random Forest.
- **Wykres 4:** Porównanie metryk dla różnych technik inżynierii cech.
- **Wykres 5:** Porównanie najlepszych modeli MLP z Random Forest.
