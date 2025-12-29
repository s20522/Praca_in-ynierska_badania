# Opisy wykresów do pracy inżynierskiej

**Autor:** [Imię Nazwisko]  
**Data:** 29 grudnia 2024  
**Rozdział:** Eksploracyjna Analiza Danych

---

## Spis wykresów

1. [Wykres 1: Rozkład zmiennej celu](#wykres-1-rozkład-zmiennej-celu)
2. [Wykres 2: Korelacje cech z DEATH_EVENT](#wykres-2-korelacje-cech-z-death_event)
3. [Wykres 3: Porównanie kluczowych cech klinicznych](#wykres-3-porównanie-kluczowych-cech-klinicznych)
4. [Wykres 4: Rozkłady kluczowych cech](#wykres-4-rozkłady-kluczowych-cech)
5. [Wykres 5: Macierz korelacji](#wykres-5-macierz-korelacji)
6. [Wykres 6: Tabela statystyk opisowych](#wykres-6-tabela-statystyk-opisowych)
7. [Wykres 7: Analiza cech binarnych](#wykres-7-analiza-cech-binarnych)
8. [Wykres 8: Analiza wartości odstających](#wykres-8-analiza-wartości-odstających)
9. [Wykres 9: Zależność wiek vs frakcja wyrzutowa](#wykres-9-zależność-wiek-vs-frakcja-wyrzutowa)
10. [Wykres 10: Zależność kreatynina vs sód](#wykres-10-zależność-kreatynina-vs-sód)

---

## Wykres 1: Rozkład zmiennej celu

**Plik:** `thesis_fig_01_target_distribution.png`

### Opis

Wykres przedstawia rozkład zmiennej celu (DEATH_EVENT) w analizowanym zbiorze danych. Zmienna ta przyjmuje wartość 0 dla pacjentów, którzy przeżyli okres obserwacji, oraz wartość 1 dla pacjentów, którzy zmarli.

### Wyniki

- **Przeżyli:** 203 pacjentów (67.89%)
- **Zmarli:** 96 pacjentów (32.11%)

### Interpretacja

Rozkład zmiennej celu wskazuje na **niezbalansowanie klas** w stosunku około 2:1. Większość pacjentów (67.89%) przeżyła okres obserwacji, podczas gdy 32.11% zmarło. Takie niezbalansowanie jest typowe dla problemów medycznych i wymaga odpowiedniego uwzględnienia w procesie modelowania, na przykład poprzez zastosowanie technik balansowania klas (SMOTE, ADASYN) lub użycie wag klas w algorytmach uczenia maszynowego.

### Podpis do pracy

> **Rysunek X.1.** Rozkład zmiennej celu (DEATH_EVENT) w zbiorze danych. Wykres słupkowy przedstawia liczbę i procent pacjentów, którzy przeżyli (zielony) oraz zmarli (czerwony) w okresie obserwacji. Widoczne jest niezbalansowanie klas w stosunku 2:1.

---

## Wykres 2: Korelacje cech z DEATH_EVENT

**Plik:** `thesis_fig_02_correlations.png`

### Opis

Wykres przedstawia współczynniki korelacji Pearsona między poszczególnymi cechami klinicznymi a zmienną celu (DEATH_EVENT). Cecha `time` została wykluczona z analizy ze względu na problem target leakage.

### Wyniki

**Najsilniejsze korelacje (dodatnie - zwiększają ryzyko):**
- Kreatynina w surowicy: r = 0.294
- Wiek: r = 0.254

**Najsilniejsze korelacje (ujemne - zmniejszają ryzyko):**
- Frakcja wyrzutowa: r = -0.269
- Sód w surowicy: r = -0.195

**Słabe korelacje:**
- Płytki krwi, kinaza kreatynowa, płeć, palenie, cukrzyca: |r| < 0.10

### Interpretacja

Analiza korelacji ujawnia, że **frakcja wyrzutowa** (r=-0.269) oraz **kreatynina w surowicy** (r=0.294) wykazują najsilniejsze związki ze zmienną celu. Ujemna korelacja frakcji wyrzutowej wskazuje, że niższa wartość tego parametru (słabsza funkcja serca) wiąże się z wyższym ryzykiem zgonu. Dodatnia korelacja kreatyniny sugeruje, że podwyższony poziom tego markera (wskazujący na problemy z funkcją nerek) zwiększa ryzyko. Wiek również wykazuje umiarkowaną dodatnią korelację (r=0.254), co potwierdza, że starsi pacjenci są bardziej narażeni na zgon.

Cechy binarne (płeć, palenie, cukrzyca) wykazują bardzo słabe korelacje liniowe, co nie wyklucza ich istotności w modelach nieliniowych lub w interakcjach z innymi zmiennymi.

### Podpis do pracy

> **Rysunek X.2.** Współczynniki korelacji Pearsona między cechami klinicznymi a zmienną celu (DEATH_EVENT). Słupki czerwone reprezentują korelacje dodatnie (zwiększające ryzyko zgonu), słupki zielone - korelacje ujemne (zmniejszające ryzyko). Cecha `time` została wykluczona ze względu na target leakage.

---

## Wykres 3: Porównanie kluczowych cech klinicznych

**Plik:** `thesis_fig_03_key_features_comparison.png`

### Opis

Wykres składa się z czterech paneli przedstawiających wykresy pudełkowe (boxplot) dla najważniejszych cech klinicznych, z podziałem na grupy pacjentów, którzy przeżyli oraz zmarli. Każdy panel zawiera również punkty danych, wartości p z testu t-Studenta oraz średnie wartości dla obu grup.

### Wyniki

| Cecha | Średnia (przeżyli) | Średnia (zmarli) | p-value | Istotność |
|-------|-------------------|------------------|---------|-----------|
| Frakcja wyrzutowa [%] | 40.27 | 33.47 | <0.001 | *** |
| Kreatynina [mg/dL] | 1.19 | 1.84 | <0.001 | *** |
| Wiek [lata] | 58.76 | 65.22 | <0.001 | *** |
| Sód [mEq/L] | 137.80 | 136.01 | 0.001 | *** |

### Interpretacja

Wszystkie cztery kluczowe cechy wykazują **istotne statystycznie różnice** między grupami (p<0.001). Pacjenci, którzy zmarli, charakteryzowali się:

1. **Niższą frakcją wyrzutową** (33.47% vs 40.27%) - różnica 6.8 punktów procentowych, co stanowi 17% redukcję. Wskazuje to na znacznie słabszą funkcję pompowania serca.

2. **Wyższym poziomem kreatyniny** (1.84 vs 1.19 mg/dL) - różnica 0.65 mg/dL, co stanowi wzrost o 55%. Sugeruje to pogorszoną funkcję nerek, która jest częstym powikłaniem niewydolności serca.

3. **Wyższym wiekiem** (65.22 vs 58.76 lat) - różnica 6.46 lat. Potwierdza to, że wiek jest istotnym czynnikiem ryzyka.

4. **Niższym poziomem sodu** (136.01 vs 137.80 mEq/L) - różnica 1.79 mEq/L. Hiponatremia (niski poziom sodu) jest znanym markerem złego rokowania w niewydolności serca.

### Podpis do pracy

> **Rysunek X.3.** Porównanie rozkładów kluczowych cech klinicznych między pacjentami, którzy przeżyli (zielony) oraz zmarli (czerwony). Wykresy pudełkowe uzupełnione są punktami danych oraz statystykami opisowymi. Wszystkie cechy wykazują istotne statystycznie różnice (p<0.001) między grupami, co potwierdza ich wartość predykcyjną.

---

## Wykres 4: Rozkłady kluczowych cech

**Plik:** `thesis_fig_04_distributions.png`

### Opis

Wykres przedstawia histogramy rozkładów czterech kluczowych cech klinicznych z podziałem na grupy pacjentów, którzy przeżyli (zielony) oraz zmarli (czerwony). Linie przerywane oznaczają średnie wartości dla każdej grupy.

### Interpretacja

**Frakcja wyrzutowa:**
- Rozkład dla pacjentów, którzy zmarli, jest przesunięty w lewo (niższe wartości)
- Znaczna część zmarłych pacjentów ma frakcję wyrzutową poniżej 30% (ciężka dysfunkcja)
- Rozkład dla pacjentów, którzy przeżyli, jest bardziej rozproszony z wyższą średnią

**Kreatynina w surowicy:**
- Rozkład dla zmarłych pacjentów jest przesunięty w prawo (wyższe wartości)
- Obecność długiego "ogona" w prawo wskazuje na przypadki z bardzo wysoką kreatyniną
- Większość pacjentów, którzy przeżyli, ma kreatynynę w normie (<1.2 mg/dL)

**Wiek:**
- Rozkład dla zmarłych pacjentów jest przesunięty w prawo (starsi pacjenci)
- Oba rozkłady są zbliżone do normalnego
- Wyraźna różnica w średnich (65 vs 59 lat)

**Sód w surowicy:**
- Rozkład dla zmarłych pacjentów jest przesunięty w lewo (niższe wartości)
- Większość wartości mieści się w normie (135-145 mEq/L)
- Zmarli pacjenci częściej mają wartości poniżej dolnej granicy normy (hiponatremia)

### Podpis do pracy

> **Rysunek X.4.** Rozkłady kluczowych cech klinicznych z podziałem na grupy: przeżyli (zielony) oraz zmarli (czerwony). Histogramy uzupełnione są liniami przerywanymi oznaczającymi średnie wartości. Widoczne są wyraźne przesunięcia rozkładów między grupami, co potwierdza różnice w charakterystykach klinicznych.

---

## Wykres 5: Macierz korelacji

**Plik:** `thesis_fig_05_correlation_heatmap.png`

### Opis

Heatmapa przedstawia macierz korelacji między czterema kluczowymi cechami klinicznymi oraz zmienną celu (DEATH_EVENT). Wartości współczynników korelacji Pearsona są przedstawione w każdej komórce.

### Wyniki kluczowych korelacji

**Między cechami:**
- Wiek ↔ Frakcja wyrzutowa: r = -0.439 (umiarkowana ujemna)
- Wiek ↔ Kreatynina: r = 0.165 (słaba dodatnia)
- Frakcja wyrzutowa ↔ Sód: r = 0.203 (słaba dodatnia)
- Kreatynina ↔ Sód: r = -0.205 (słaba ujemna)

**Z DEATH_EVENT:**
- Frakcja wyrzutowa: r = -0.269 (najsilniejsza)
- Kreatynina: r = 0.294 (najsilniejsza)
- Wiek: r = 0.254
- Sód: r = -0.195

### Interpretacja

Macierz korelacji ujawnia kilka istotnych zależności:

1. **Ujemna korelacja wieku z frakcją wyrzutową** (r=-0.439) sugeruje, że starsi pacjenci mają tendencję do niższej frakcji wyrzutowej, co jest zgodne z naturalnym procesem starzenia się serca.

2. **Słabe korelacje między większością cech** wskazują, że są one względnie niezależne i wnoszą komplementarną informację do modeli predykcyjnych.

3. **Brak silnych korelacji między cechami** (wszystkie |r| < 0.5) sugeruje, że nie ma problemu z multikolinearością, co jest korzystne dla modelowania.

### Podpis do pracy

> **Rysunek X.5.** Macierz korelacji między kluczowymi cechami klinicznymi a zmienną celu (DEATH_EVENT). Wartości współczynników korelacji Pearsona przedstawione są w skali kolorów od niebieskiego (korelacja ujemna) przez biały (brak korelacji) do czerwonego (korelacja dodatnia). Brak silnych korelacji między cechami wskazuje na ich względną niezależność.

---

## Wykres 6: Tabela statystyk opisowych

**Plik:** `thesis_fig_06_statistics_table.png`

### Opis

Tabela przedstawia szczegółowe statystyki opisowe dla czterech kluczowych cech klinicznych, obejmujące całą populację oraz osobno dla grup pacjentów, którzy przeżyli i zmarli.

### Interpretacja

Tabela dostarcza kompleksowego przeglądu charakterystyk danych:

**Frakcja wyrzutowa:**
- Duża zmienność (odch. std. = 11.83)
- Zakres od 14% (ciężka dysfunkcja) do 80% (prawie norma)
- Mediana (38%) zbliżona do średniej (38.08%), co wskazuje na rozkład symetryczny

**Kreatynina w surowicy:**
- Wysoka zmienność (odch. std. = 1.03)
- Wartość maksymalna (9.40 mg/dL) znacznie przekracza normę, wskazując na przypadki ostrej niewydolności nerek
- Mediana (1.10) niższa od średniej (1.39), co sugeruje rozkład prawostronnie skośny

**Wiek:**
- Umiarkowana zmienność (odch. std. = 11.89)
- Zakres 40-95 lat obejmuje szeroki przedział wiekowy
- Mediana (60) zbliżona do średniej (60.83)

**Sód w surowicy:**
- Niska zmienność (odch. std. = 4.41)
- Większość wartości w normie (135-145 mEq/L)
- Wartość minimalna (114 mEq/L) wskazuje na przypadki ciężkiej hiponatremii

### Podpis do pracy

> **Tabela X.1.** Statystyki opisowe kluczowych cech klinicznych. Przedstawiono średnią, medianę, odchylenie standardowe, wartości minimalne i maksymalne dla całej populacji oraz średnie wartości osobno dla grup pacjentów, którzy przeżyli i zmarli. Widoczne są istotne różnice w średnich między grupami.

---

## Wykres 7: Analiza cech binarnych

**Plik:** `thesis_fig_07_binary_features.png`

### Opis

Wykres składa się z pięciu paneli przedstawiających rozkłady cech binarnych (anemia, nadciśnienie, cukrzyca, palenie, płeć) z podziałem na grupy pacjentów, którzy przeżyli oraz zmarli. Każdy panel zawiera wyniki testu chi-kwadrat.

### Wyniki testów chi-kwadrat

| Cecha | χ² | p-value | Istotność | Interpretacja |
|-------|-----|---------|-----------|---------------|
| Anemia | 5.89 | 0.015 | * | Istotny związek |
| Nadciśnienie | 4.12 | 0.042 | * | Istotny związek |
| Cukrzyca | 0.00 | 0.974 | ns | Brak związku |
| Palenie | 0.03 | 0.869 | ns | Brak związku |
| Płeć | 0.00 | 0.995 | ns | Brak związku |

### Interpretacja

**Anemia:**
- Istotny statystycznie związek z DEATH_EVENT (p=0.015)
- Pacjenci z anemią mają wyższe ryzyko zgonu
- Anemia jest znanym czynnikiem ryzyka w niewydolności serca, ponieważ pogarsza dostarczanie tlenu do tkanek

**Nadciśnienie:**
- Istotny statystycznie związek z DEATH_EVENT (p=0.042)
- Pacjenci z nadciśnieniem mają wyższe ryzyko zgonu
- Nadciśnienie zwiększa obciążenie serca i może przyspieszać progresję niewydolności

**Cukrzyca, palenie, płeć:**
- Brak istotnych statystycznie związków w analizie univariate
- Nie oznacza to, że są nieistotne - mogą być ważne w modelach multivariate lub w interakcjach z innymi zmiennymi

### Podpis do pracy

> **Rysunek X.7.** Analiza cech binarnych w kontekście DEATH_EVENT. Wykresy słupkowe zgrupowane przedstawiają liczbę pacjentów w każdej kategorii z podziałem na grupy: przeżyli (zielony) oraz zmarli (czerwony). Adnotacje zawierają wyniki testów chi-kwadrat. Anemia i nadciśnienie wykazują istotne statystycznie związki ze zmienną celu (p<0.05).

---

## Wykres 8: Analiza wartości odstających

**Plik:** `thesis_fig_08_outliers.png`

### Opis

Wykres przedstawia wykresy pudełkowe (boxplot) dla sześciu cech numerycznych, służące do identyfikacji wartości odstających metodą IQR (Interquartile Range). Czerwone punkty oznaczają wartości odstające.

### Wyniki

| Cecha | Liczba outlierów | % outlierów | Zakres IQR |
|-------|------------------|-------------|------------|
| Wiek | 0 | 0.00% | [22.5, 98.5] |
| Frakcja wyrzutowa | 2 | 0.67% | [7.5, 67.5] |
| Kreatynina | 29 | 9.70% | [0.15, 2.15] |
| Sód | 4 | 1.34% | [125.0, 149.0] |
| Płytki krwi | 21 | 7.02% | [76k, 440k] |
| Kinaza kreatynowa | 29 | 9.70% | [-582, 1280] |

### Interpretacja

**Cechy z dużą liczbą outlierów:**
- **Kreatynina w surowicy** (9.70%): Wartości odstające reprezentują przypadki z ciężką niewydolnością nerek. Nie należy ich usuwać, ponieważ są klinicznie istotne.
- **Kinaza kreatynowa** (9.70%): Wysokie wartości mogą wskazywać na uszkodzenie mięśnia sercowego lub szkieletowego.
- **Płytki krwi** (7.02%): Zarówno bardzo niskie (trombocytopenia), jak i wysokie wartości mogą mieć znaczenie kliniczne.

**Rekomendacje:**
- Wartości odstające **nie powinny być automatycznie usuwane**, ponieważ mogą reprezentować rzeczywiste przypadki kliniczne o wysokim ryzyku
- Należy rozważyć zastosowanie robust scaling (np. RobustScaler) zamiast standardowej standaryzacji
- W modelach drzewiastych (Random Forest, XGBoost) wartości odstające nie stanowią problemu

### Podpis do pracy

> **Rysunek X.8.** Analiza wartości odstających metodą IQR dla cech numerycznych. Wykresy pudełkowe przedstawiają rozkład wartości, gdzie czerwone punkty oznaczają wartości odstające (poza zakresem Q1-1.5×IQR, Q3+1.5×IQR). Adnotacje zawierają liczbę i procent wartości odstających oraz zakres IQR. Największą liczbę outlierów obserwuje się w kreatyninie (9.70%) i kinazie kreatynowej (9.70%).

---

## Wykres 9: Zależność wiek vs frakcja wyrzutowa

**Plik:** `thesis_fig_09_age_vs_ef.png`

### Opis

Wykres punktowy (scatter plot) przedstawia zależność między wiekiem pacjenta a frakcją wyrzutową serca, z podziałem na grupy pacjentów, którzy przeżyli (zielony) oraz zmarli (czerwony). Linie przerywane oznaczają progi kliniczne.

### Interpretacja

**Obserwowane wzorce:**

1. **Ujemna korelacja wiek-frakcja wyrzutowa:** Widoczny jest trend spadkowy - starsi pacjenci mają tendencję do niższej frakcji wyrzutowej, co jest zgodne z naturalnym procesem starzenia się serca.

2. **Skupisko zmarłych pacjentów w lewym dolnym rogu:** Pacjenci, którzy zmarli, koncentrują się w obszarze niskiej frakcji wyrzutowej (<30%) i wyższego wieku (>60 lat), co wskazuje na szczególnie wysokie ryzyko w tej grupie.

3. **Progi kliniczne:**
   - **Linia czerwona (30%):** Próg ciężkiej dysfunkcji skurczowej. Pacjenci poniżej tego progu mają znacznie wyższe ryzyko zgonu.
   - **Linia zielona (50%):** Dolna granica normy. Wartości powyżej 50% uważane są za prawidłowe.

4. **Strefa wysokiego ryzyka:** Obszar poniżej 30% frakcji wyrzutowej i powyżej 60 lat wieku charakteryzuje się najwyższą śmiertelnością.

### Podpis do pracy

> **Rysunek X.9.** Zależność między wiekiem a frakcją wyrzutową z uwzględnieniem DEATH_EVENT. Wykres punktowy przedstawia poszczególnych pacjentów: przeżyli (zielony) oraz zmarli (czerwony). Linie przerywane oznaczają progi kliniczne: 30% (ciężka dysfunkcja) i 50% (dolna granica normy). Widoczna jest ujemna korelacja oraz skupisko zmarłych pacjentów w obszarze niskiej frakcji wyrzutowej i wyższego wieku.

---

## Wykres 10: Zależność kreatynina vs sód

**Plik:** `thesis_fig_10_creatinine_vs_sodium.png`

### Opis

Wykres punktowy przedstawia zależność między poziomem kreatyniny w surowicy a poziomem sodu w surowicy, z podziałem na grupy pacjentów, którzy przeżyli (zielony) oraz zmarli (czerwony). Linie przerywane oznaczają progi kliniczne.

### Interpretacja

**Obserwowane wzorce:**

1. **Skupisko zmarłych pacjentów w prawym dolnym rogu:** Pacjenci, którzy zmarli, koncentrują się w obszarze wysokiej kreatyniny (>1.2 mg/dL) i niskiego sodu (<135 mEq/L), co wskazuje na szczególnie wysokie ryzyko w tej grupie.

2. **Progi kliniczne:**
   - **Linia czerwona pionowa (1.2 mg/dL):** Górna granica normy dla kreatyniny. Wartości powyżej wskazują na pogorszoną funkcję nerek.
   - **Linia pomarańczowa pozioma (135 mEq/L):** Dolna granica normy dla sodu. Wartości poniżej definiują hiponatremię.

3. **Strefa wysokiego ryzyka:** Obszar wysokiej kreatyniny (>1.2 mg/dL) i niskiego sodu (<135 mEq/L) charakteryzuje się najwyższą śmiertelnością. Kombinacja niewydolności nerek i zaburzeń elektrolitowych jest szczególnie niekorzystna prognostycznie.

4. **Słaba korelacja między cechami:** Brak wyraźnego wzorca liniowego między kreatyniną a sodem (r=-0.205) sugeruje, że są to względnie niezależne markery, wnoszące komplementarną informację prognostyczną.

### Podpis do pracy

> **Rysunek X.10.** Zależność między kreatyniną w surowicy a sodem w surowicy z uwzględnieniem DEATH_EVENT. Wykres punktowy przedstawia poszczególnych pacjentów: przeżyli (zielony) oraz zmarli (czerwony). Linie przerywane oznaczają progi kliniczne: 1.2 mg/dL (górna granica normy kreatyniny) i 135 mEq/L (dolna granica normy sodu). Widoczne jest skupisko zmarłych pacjentów w obszarze wysokiej kreatyniny i niskiego sodu.

---

## Podsumowanie

Wygenerowano **10 profesjonalnych wykresów** w rozdzielczości 300 DPI, odpowiednich do umieszczenia w pracy inżynierskiej. Wszystkie wykresy zawierają:

- ✅ Polskie opisy osi i tytułów
- ✅ Czytelne legendy
- ✅ Odpowiednie kolory i style
- ✅ Statystyki i adnotacje
- ✅ Wysoką jakość graficzną

### Rekomendacje do użycia w pracy:

1. **Wykresy 1-2:** Rozdział "Eksploracyjna Analiza Danych" - podstawowa charakterystyka
2. **Wykresy 3-5:** Rozdział "Analiza cech klinicznych" - szczegółowa analiza
3. **Wykres 6:** Rozdział "Statystyki opisowe" - tabela podsumowująca
4. **Wykres 7:** Rozdział "Analiza cech kategorycznych"
5. **Wykres 8:** Rozdział "Preprocessing i jakość danych"
6. **Wykresy 9-10:** Rozdział "Analiza zależności między cechami"

### Format cytowania wykresów:

```
Jak przedstawiono na Rysunku X.Y, rozkład zmiennej celu wskazuje na 
niezbalansowanie klas w stosunku 2:1 (67.89% przeżyło, 32.11% zmarło).
```

---

**Autor:** Heart Failure Research Team  
**Data:** 29 grudnia 2024  
**Wersja:** 1.0
