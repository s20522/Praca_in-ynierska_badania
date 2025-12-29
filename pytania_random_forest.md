# 🎓 Pytania do obrony pracy inżynierskiej - Model Random Forest

**Autor:** Heart Failure Research Team  
**Data:** 29 grudnia 2024  
**Cel:** Przygotowanie do obrony pracy inżynierskiej poprzez zrozumienie kluczowych aspektów budowy, optymalizacji i ewaluacji modelu Random Forest.

---

## Wprowadzenie

Poniższa lista 30 pytań (wraz z odpowiedziami) została stworzona, aby pomóc w przygotowaniu się do obrony pracy inżynierskiej w sekcji dotyczącej modelu Random Forest. Pytania obejmują zarówno podstawy teoretyczne algorytmu, jak i szczegółową interpretację wyników uzyskanych w trakcie eksperymentów. Celem jest nie tylko "wykucie" odpowiedzi, ale przede wszystkim **głębokie zrozumienie** materiału.

## Spis Kategorii

1.  [**I. Teoria i Działanie Modelu (Pytania 1-7)**](#i-teoria-i-działanie-modelu)
2.  [**II. Przygotowanie Danych i Budowa Modelu (Pytania 8-14)**](#ii-przygotowanie-danych-i-budowa-modelu)
3.  [**III. Optymalizacja i Walidacja (Pytania 15-19)**](#iii-optymalizacja-i-walidacja)
4.  [**IV. Ewaluacja i Metryki (Pytania 20-26)**](#iv-ewaluacja-i-metryki)
5.  [**V. Interpretacja Wyników i Wnioski (Pytania 27-30)**](#v-interpretacja-wyników-i-wnioski)

---

## I. Teoria i Działanie Modelu

### Pytanie 1: Proszę wyjaśnić, czym jest model Random Forest i na czym polega jego działanie?

> **Odpowiedź:**
> Random Forest (Las Losowy) to **zespołowy model uczenia maszynowego**, który składa się z wielu pojedynczych drzew decyzyjnych. Jego działanie opiera się na zasadzie "mądrości tłumu". W procesie klasyfikacji, każde drzewo w "lesie" oddaje swój "głos" na daną klasę, a ostateczna predykcja modelu to klasa, która otrzymała najwięcej głosów. Taka agregacja wyników z wielu różnych drzew sprawia, że model jest znacznie bardziej stabilny i odporny na błędy niż pojedyncze drzewo decyzyjne.

### Pytanie 2: Na czym polega "losowość" w modelu Random Forest? Proszę wymienić dwa główne źródła losowości.

> **Odpowiedź:**
> "Losowość" jest kluczowym elementem, który sprawia, że drzewa w lesie są zróżnicowane. Istnieją dwa główne źródła losowości:
> 1.  **Bootstrap Aggregating (Bagging):** Każde drzewo jest trenowane na nieco innym podzbiorze danych treningowych, stworzonym przez **losowanie ze zwracaniem**. Oznacza to, że niektóre próbki mogą pojawić się w podzbiorze wielokrotnie, a inne wcale.
> 2.  **Losowy wybór cech:** Podczas budowy każdego węzła w drzewie, algorytm nie rozważa wszystkich dostępnych cech, ale tylko ich **losowo wybrany podzbiór**. To zmusza drzewa do szukania różnych zależności i zapobiega dominacji jednej, silnej cechy.

### Pytanie 3: Jakie są główne zalety stosowania modelu Random Forest w porównaniu do pojedynczego drzewa decyzyjnego?

> **Odpowiedź:**
> Główne zalety to:
> - **Redukcja przeuczenia (overfitting):** Pojedyncze drzewa mają tendencję do "uczenia się na pamięć" danych treningowych. Agregacja wyników z wielu drzew uśrednia błędy i sprawia, że model lepiej generalizuje na nowe dane.
> - **Większa stabilność i dokładność:** Model jest mniej wrażliwy na niewielkie zmiany w danych treningowych. Wynik oparty na "głosowaniu" setek drzew jest bardziej wiarygodny.
> - **Wbudowana ocena ważności cech:** Algorytm potrafi ocenić, które cechy miały największy wkład w proces decyzyjny.

### Pytanie 4: Czy model Random Forest jest wrażliwy na skalę cech? Czy wymaga normalizacji/standaryzacji?

> **Odpowiedź:**
> Nie, modele oparte na drzewach decyzyjnych, w tym Random Forest, **nie są wrażliwe na skalę cech**. Działają one na zasadzie podziałów (np. "czy wiek > 60?"), a nie na odległościach, jak np. SVM czy sieci neuronowe. Dlatego standaryzacja **nie jest technicznie wymagana** do poprawnego działania algorytmu. W moim projekcie zastosowałem ją jednak dla spójności z innymi modelami, które planuję testować w przyszłości, oraz jako dobrą praktykę w budowaniu potoków uczenia maszynowego.

### Pytanie 5: Co to jest "out-of-bag error" i jak można go wykorzystać?

> **Odpowiedź:**
> Dzięki technice baggingu, każde drzewo jest trenowane tylko na części danych (średnio ok. 2/3). Pozostała 1/3 próbek, które nie zostały użyte do treningu danego drzewa, nazywana jest **próbkami "out-of-bag" (OOB)**. Możemy wykorzystać te próbki do oceny modelu bez potrzeby tworzenia osobnego zbioru walidacyjnego. Dla każdej próbki zbieramy "głosy" tylko od tych drzew, które nie widziały jej podczas treningu. Błąd popełniony na tych próbkach to właśnie **"out-of-bag error"**, który jest dobrym i bezstronnym estymatorem błędu generalizacji.

### Pytanie 6: Jak Random Forest radzi sobie z problemem niezbalansowanych klas?

> **Odpowiedź:**
> Standardowy Random Forest może mieć tendencję do faworyzowania klasy większościowej. Jednak algorytm oferuje skuteczne mechanizmy radzenia sobie z tym problemem. W moim projekcie wykorzystałem hiperparametr `class_weight=\'balanced\'`. Ta opcja automatycznie dostosowuje wagi klas odwrotnie proporcjonalnie do ich liczebności. Oznacza to, że błędy popełnione na klasie mniejszościowej (w naszym przypadku `DEATH_EVENT = 1`) są "karane" znacznie surowiej, co zmusza model do zwracania na nie większej uwagi.

### Pytanie 7: Czym różni się Random Forest od innych modeli zespołowych, np. Gradient Boosting (XGBoost)?

> **Odpowiedź:**
> Główna różnica leży w sposobie budowania zespołu. W **Random Forest** drzewa są budowane **niezależnie i równolegle**. Każde drzewo jest "ekspertem" od czegoś innego, a na końcu odbywa się głosowanie. W **Gradient Boosting** drzewa są budowane **sekwencyjnie**. Każde kolejne drzewo uczy się na błędach popełnionych przez poprzednie. Jest to proces iteracyjny, gdzie model stopniowo "poprawia" swoje słabości. W efekcie Gradient Boosting często osiąga nieco wyższą dokładność, ale jest bardziej podatny na przeuczenie i wymaga staranniejszej optymalizacji.

---

## II. Przygotowanie Danych i Budowa Modelu

### Pytanie 8: Dlaczego do budowy modelu wybrano tylko trzy cechy: wiek, frakcja wyrzutowa i kreatynina w surowicy?

> **Odpowiedź:**
> Decyzja ta opierała się na dwóch filarach: po pierwsze, na wynikach mojej **eksploracyjnej analizy danych (EDA)**, która wykazała, że te trzy cechy mają najsilniejszą korelację ze zmienną celu. Po drugie, jest to zgodne z **metodyką z publikacji bazowej**, gdzie autorzy, na podstawie analizy przeżycia Coksa, również zidentyfikowali te cechy jako najważniejsze predyktory. Skupienie się na najsilniejszych sygnałach pozwala zbudować prostszy, bardziej interpretowalny i często bardziej stabilny model.

### Pytanie 9: Jaki był cel podziału danych na zbiór treningowy i testowy?

> **Odpowiedź:**
> Celem tego podziału jest **rzetelna ocena zdolności generalizacji modelu**. Zbiór treningowy (80% danych) służy do "nauczenia" modelu wzorców. Zbiór testowy (20% danych) jest trzymany "w ukryciu" i używany dopiero na samym końcu. Ocena na danych, których model nigdy wcześniej nie widział, symuluje jego działanie w rzeczywistych warunkach i pozwala sprawdzić, czy nie nauczył się on po prostu na pamięć zbioru treningowego (overfitting).

### Pytanie 10: Co to jest stratyfikacja i dlaczego została użyta podczas podziału danych?

> **Odpowiedź:**
> **Stratyfikacja** to technika, która zapewnia, że proporcje klas w zbiorze danych są zachowane po podziale. W naszym przypadku, klasa `DEATH_EVENT` jest niezbalansowana (68% przeżyło, 32% zmarło). Bez stratyfikacji mogłoby się zdarzyć, że w wyniku losowego podziału, w zbiorze testowym znalazłoby się np. bardzo mało przypadków zgonów, co sprawiłoby, że ocena modelu byłaby niewiarygodna. Dzięki stratyfikacji, zarówno w zbiorze treningowym, jak i testowym, mamy taki sam procentowy rozkład klas jak w oryginalnym zbiorze.

### Pytanie 11: Jakie konkretne kroki obejmował potok (pipeline) uczenia maszynowego w Pana/Pani projekcie?

> **Odpowiedź:**
> Mój potok składał się z następujących kroków:
> 1.  **Wczytanie i selekcja cech:** Wczytanie danych i wybór trzech kluczowych predyktorów.
> 2.  **Podział danych:** Podział na zbiór treningowy (80%) i testowy (20%) z stratyfikacją.
> 3.  **Standaryzacja:** Dopasowanie `StandardScaler` na zbiorze treningowym i transformacja obu zbiorów.
> 4.  **Optymalizacja hiperparametrów:** Użycie `RandomizedSearchCV` z 5-krotną walidacją krzyżową na zbiorze treningowym w celu znalezienia najlepszych parametrów dla modelu.
> 5.  **Trening finalnego modelu:** Wytrenowanie modelu Random Forest z optymalnymi parametrami na całym zbiorze treningowym.
> 6.  **Ewaluacja:** Ocena modelu na zbiorze testowym za pomocą różnych metryk (F1, Recall, AUC itp.).

### Pytanie 12: Jakie były wymiary zbioru treningowego i testowego?

> **Odpowiedź:**
> Oryginalny zbiór liczył 299 próbek. Po podziale 80/20, zbiór treningowy składał się z **239 próbek**, a zbiór testowy z **60 próbek**.

### Pytanie 13: Czy model był trenowany na danych surowych czy przeskalowanych?

> **Odpowiedź:**
> Model był trenowany na danych **przeskalowanych** za pomocą `StandardScaler`. Chociaż Random Forest nie wymaga skalowania, jest to dobra praktyka, która zapewnia spójność i ułatwia porównywanie z innymi modelami, które planuję testować, a które są wrażliwe na skalę cech (np. sieci neuronowe).

### Pytanie 14: Jakie oprogramowanie i biblioteki zostały wykorzystane do implementacji modelu?

> **Odpowiedź:**
> Cały proces został zaimplementowany w języku **Python** przy użyciu biblioteki **Scikit-learn**. `RandomForestClassifier` posłużył do budowy modelu, `train_test_split` do podziału danych, `StandardScaler` do normalizacji, a `RandomizedSearchCV` do optymalizacji hiperparametrów. Do wizualizacji wyników użyłem bibliotek **Matplotlib** i **Seaborn**.

---

## III. Optymalizacja i Walidacja

### Pytanie 15: Co to są hiperparametry i dlaczego ich optymalizacja jest ważna?

> **Odpowiedź:**
> Hiperparametry to zewnętrzne parametry konfiguracyjne modelu, które nie są uczone z danych, ale muszą być ustawione przed treningiem. Przykłady to liczba drzew w lesie czy maksymalna głębokość drzewa. Ich optymalizacja jest kluczowa, ponieważ odpowiednie wartości mogą znacząco **poprawić skuteczność modelu**, podczas gdy złe ustawienia mogą prowadzić do przeuczenia lub niedouczenia. Celem optymalizacji jest znalezienie "złotego środka", który zapewni najlepszą zdolność generalizacji.

### Pytanie 16: Czym różni się `RandomizedSearchCV` od `GridSearchCV` i dlaczego wybrano to pierwsze?

> **Odpowiedź:**
> `GridSearchCV` testuje **wszystkie możliwe kombinacje** zdefiniowanych hiperparametrów. Jest to bardzo dokładne, ale przy dużej liczbie parametrów staje się niezwykle czasochłonne. `RandomizedSearchCV` testuje tylko **losowo wybraną liczbę kombinacji** (w moim przypadku 100). Wybrałem `RandomizedSearchCV`, ponieważ jest znacznie szybszy i często znajduje równie dobre (lub prawie tak dobre) wyniki, co `GridSearchCV`, zwłaszcza gdy niektóre hiperparametry mają mniejszy wpływ na wynik. Pozwoliło mi to przeszukać szerszą przestrzeń parametrów w rozsądnym czasie.

### Pytanie 17: Co to jest walidacja krzyżowa (cross-validation) i dlaczego jest lepsza niż pojedynczy podział walidacyjny?

> **Odpowiedź:**
> Walidacja krzyżowa to technika oceny modelu, która polega na wielokrotnym podziale danych na zbiory treningowe i walidacyjne. W **k-krotnej walidacji krzyżowej** (u mnie k=5), dane są dzielone na k części. Model jest trenowany k razy, za każdym razem na k-1 częściach, a testowany na pozostałej. Wyniki są uśredniane. Jest to metoda znacznie bardziej **wiarygodna i stabilna** niż pojedynczy podział, ponieważ ocena nie zależy od "szczęścia" w losowym podziale danych. Daje nam lepsze pojęcie o tym, jak model będzie się zachowywał na zupełnie nowych danych.

### Pytanie 18: Jakie były najlepsze hiperparametry znalezione dla Pana/Pani modelu?

> **Odpowiedź:**
> Najlepsze znalezione hiperparametry to: 100 drzew (`n_estimators`), minimalna liczba próbek do podziału równa 5 (`min_samples_split`), minimalna liczba próbek w liściu równa 8 (`min_samples_leaf`) oraz, co bardzo ważne, użycie wag klas (`class_weight=\'balanced\'`). Ustawienie `min_samples_leaf` na 8 pomaga w regularyzacji, zapobiegając tworzeniu zbyt skomplikowanych drzew, które mogłyby się przeuczyć.

### Pytanie 19: Jaką metrykę wybrano do optymalizacji w `RandomizedSearchCV` i dlaczego?

> **Odpowiedź:**
> Do optymalizacji wybrano metrykę **F1-score**. W przypadku niezbalansowanych danych, optymalizacja pod kątem samej dokładności (accuracy) byłaby błędem, ponieważ model mógłby osiągnąć wysoki wynik, ignorując klasę mniejszościową. F1-score jest średnią harmoniczną precyzji i czułości, co czyni go dobrym, zbalansowanym wskaźnikiem skuteczności, gdy zależy nam na poprawnym klasyfikowaniu obu klas.

---

## IV. Ewaluacja i Metryki

### Pytanie 20: Proszę wyjaśnić, co oznaczają metryki: Precision, Recall i F1-score.

> **Odpowiedź:**
> - **Precision (Precyzja):** Mówi nam, **jaki procent predykcji pozytywnych był poprawny**. W naszym kontekście: "Spośród wszystkich pacjentów, których model oznaczył jako zagrożonych zgonem, ilu faktycznie zmarło?". Wysoka precyzja oznacza mało fałszywych alarmów.
> - **Recall (Czułość):** Mówi nam, **jaki procent wszystkich faktycznych przypadków pozytywnych model poprawnie wykrył**. W naszym kontekście: "Spośród wszystkich pacjentów, którzy faktycznie zmarli, ilu model poprawnie zidentyfikował?". Wysoka czułość oznacza mało "przegapionych" przypadków.
> - **F1-score:** Jest to **średnia harmoniczna precyzji i czułości**. Stanowi kompromis między tymi dwiema metrykami i jest szczególnie użyteczna, gdy mamy do czynienia z niezbalansowanymi klasami.

### Pytanie 21: W Pana/Pani modelu Recall (89.5%) jest znacznie wyższy niż Precision (54.8%). Co to oznacza w kontekście medycznym i czy jest to pożądany wynik?

> **Odpowiedź:**
> Tak, w tym konkretnym problemie medycznym jest to **bardzo pożądany wynik**. Wysoki Recall oznacza, że nasz model jest **bardzo skuteczny w identyfikowaniu pacjentów, którzy faktycznie są w grupie wysokiego ryzyka** (przegapił tylko 2 z 19 takich pacjentów w zbiorze testowym). Niższa precyzja oznacza, że model generuje pewną liczbę "fałszywych alarmów" (14 pacjentów oznaczonych jako zagrożeni, którzy przeżyli). Z klinicznego punktu widzenia, konsekwencje **przegapienia pacjenta wysokiego ryzyka (False Negative)** są znacznie poważniejsze niż konsekwencje **fałszywego alarmu (False Positive)**, który co najwyżej skieruje pacjenta na dodatkowe badania. Dlatego priorytetyzacja Recall jest tutaj uzasadniona.

### Pytanie 22: Co przedstawia macierz pomyłek (confusion matrix) i jak ją interpretować?

> **Odpowiedź:**
> Macierz pomyłek to tabela, która podsumowuje wyniki klasyfikacji. Wiersze reprezentują prawdziwe klasy, a kolumny - klasy przewidziane przez model. W naszej macierzy:
> - **True Positives (TP=17):** Poprawnie zidentyfikowani zmarli.
> - **True Negatives (TN=27):** Poprawnie zidentyfikowani żyjący.
> - **False Positives (FP=14):** Żyjący, błędnie oznaczeni jako zmarli ("fałszywy alarm").
> - **False Negatives (FN=2):** Zmarli, błędnie oznaczeni jako żyjący ("przegapienie").

Macierz ta daje pełny obraz działania modelu, znacznie bardziej szczegółowy niż pojedyncza metryka jak dokładność.

### Pytanie 23: Co to jest krzywa ROC i co oznacza pole pod nią (AUC)?

> **Odpowiedź:**
> Krzywa ROC (Receiver Operating Characteristic) to wykres, który pokazuje zdolność modelu do odróżniania klas. Ilustruje ona kompromis między **True Positive Rate (Recall)** a **False Positive Rate** przy różnych progach decyzyjnych. **Pole pod krzywą (AUC - Area Under the Curve)** jest pojedynczą liczbą podsumowującą jej jakość. AUC równe 1.0 oznacza idealny klasyfikator, a 0.5 - klasyfikator losowy. Nasz wynik **AUC = 0.7689** oznacza, że model ma dobrą, znacznie lepszą od losowej, zdolność do rozróżniania pacjentów, którzy przeżyją, od tych, którzy umrą.

### Pytanie 24: Dlaczego oprócz krzywej ROC analizuje się również krzywą Precision-Recall?

> **Odpowiedź:**
> Krzywa ROC może być zbyt optymistyczna w przypadku silnie niezbalansowanych danych, ponieważ uwzględnia True Negatives, których jest bardzo dużo. Krzywa Precision-Recall, która pokazuje kompromis między precyzją a czułością, jest często uważana za bardziej informatywną w takich przypadkach. Skupia się ona na wydajności modelu w odniesieniu do rzadkiej, pozytywnej klasy, co jest kluczowe w naszym problemie. Nasz wynik **AUC-PR = 0.6961**, znacznie powyżej linii bazowej, potwierdza, że model jest skuteczny.

### Pytanie 25: Wyniki z walidacji krzyżowej (CV F1-score: 0.6661 ± 0.0857) są nieco niższe niż na zbiorze testowym (F1-score: 0.6800). Co to oznacza?

> **Odpowiedź:**
> Wyniki są bardzo zbliżone, co jest dobrym znakiem. Walidacja krzyżowa daje bardziej uśrednioną i realistyczną ocenę wydajności modelu. Niewielkie odchylenie standardowe (±0.0857) świadczy o tym, że model jest **stabilny** i jego wyniki nie zależą mocno od konkretnego podziału danych. Fakt, że wynik na zbiorze testowym jest podobny do średniej z CV, potwierdza, że nasz model dobrze generalizuje i nie jest przeuczony.

### Pytanie 26: Czy uzyskane wyniki są satysfakcjonujące w porównaniu do wyników z publikacji bazowej?

> **Odpowiedź:**
> Publikacja bazowa podaje F1-score na poziomie 88.37% dla modelu SVM, podczas gdy nasz Random Forest osiągnął 68.00%. Różnica ta może wynikać z kilku czynników: użyliśmy tylko 3 cech, podczas gdy autorzy mogli użyć ich więcej; mogli też zastosować inną metodologię preprocessingu. Należy jednak podkreślić, że naszym celem była **reprodukcja metodyki i zbudowanie klinicznie użytecznego modelu**, a nie ślepe dążenie do jak najwyższego wyniku. Nasz model, z **Recall na poziomie 89.5%**, jest bardzo wartościowy z praktycznego punktu widzenia, nawet jeśli jego F1-score jest niższy.

---

## V. Interpretacja Wyników i Wnioski

### Pytanie 27: Co oznacza mechanizm "feature importance" w Random Forest i jakie były jego wyniki w Pana/Pani modelu?

> **Odpowiedź:**
> "Feature importance" to mechanizm, który ocenia, jak bardzo każda cecha przyczyniła się do poprawy czystości podziałów (redukcji zanieczyszczenia) we wszystkich drzewach w lesie. Innymi słowy, mówi nam, które cechy były najczęściej i najskuteczniej wykorzystywane przez model do podejmowania decyzji. W moim modelu wyniki były jednoznaczne:
> 1.  **Kreatynina w surowicy (46.07%):** Zdecydowanie najważniejsza cecha.
> 2.  **Frakcja wyrzutowa (33.76%):** Druga co do ważności.
> 3.  **Wiek (20.17%):** Trzecia, ale wciąż istotna.

### Pytanie 28: Jakie praktyczne, kliniczne wnioski można wyciągnąć z faktu, że kreatynina w surowicy jest najważniejszą cechą?

> **Odpowiedź:**
> Wniosek jest taki, że **stan nerek jest kluczowym czynnikiem prognostycznym** u pacjentów z niewydolnością serca. Potwierdza to istnienie tzw. **zespołu sercowo-nerkowego**, gdzie niewydolność jednego organu napędza niewydolność drugiego. Z praktycznego punktu widzenia oznacza to, że monitorowanie funkcji nerek (poprzez pomiar kreatyniny) powinno być absolutnym priorytetem w opiece nad tymi pacjentami, a każdy wzrost jej poziomu powinien być traktowany jako poważny sygnał alarmowy.

### Pytanie 29: Czy zbudowany model mógłby być użyty w praktyce klinicznej? Jakie są jego ograniczenia?

> **Odpowiedź:**
> Model wykazuje duży potencjał jako **narzędzie wspomagające decyzje kliniczne**, głównie dzięki bardzo wysokiej czułości (Recall). Mógłby służyć jako system wczesnego ostrzegania, identyfikujący pacjentów, którzy wymagają pilniejszej uwagi lub intensywniejszego leczenia. Główne ograniczenia to:
> - **Mały zbiór danych:** Model został wytrenowany na tylko 299 pacjentach, co ogranicza jego zdolność generalizacji. Wymaga walidacji na znacznie większej i bardziej zróżnicowanej populacji.
> - **Ograniczona liczba cech:** Użycie tylko 3 cech upraszcza model, ale może pomijać inne istotne czynniki.
> - **Brak walidacji zewnętrznej:** Model nie był testowany na danych z innego szpitala czy kraju.

### Pytanie 30: Jakie są Pana/Pani rekomendacje dotyczące dalszych kroków w rozwijaniu tego modelu?

> **Odpowiedź:**
> Rekomenduję następujące kroki:
> 1.  **Feature Engineering:** Stworzenie nowych cech (np. wskaźnika sercowo-nerkowego) i przetestowanie, czy dodanie innych cech z oryginalnego zbioru (np. sodu, ciśnienia krwi) poprawi wyniki.
> 2.  **Testowanie innych modeli:** Porównanie Random Forest z innymi algorytmami, takimi jak XGBoost, LightGBM czy SVM, aby znaleźć najskuteczniejsze podejście.
> 3.  **Budowa modeli sieci neuronowych (MLP):** Sprawdzenie, czy bardziej złożone modele są w stanie uchwycić nieliniowe zależności, których Random Forest mógł nie dostrzec.
> 4.  **Analiza przeżycia:** Zastosowanie modeli specyficznych dla analizy przeżycia (np. DeepSurv), które modelują nie tylko to, CZY pacjent umrze, ale również KIEDY to może nastąpić.
