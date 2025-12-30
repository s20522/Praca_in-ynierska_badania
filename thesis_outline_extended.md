# Rozszerzony Szkielet Pracy Inżynierskiej: Predykcja Niewydolności Serca

**Tytuł:** *Zastosowanie modeli uczenia maszynowego, w tym sieci neuronowych, do predykcji śmiertelności pacjentów z niewydolnością serca.*

**Autor:** [Twoje Imię i Nazwisko]

**Repozytorium z kodem:** [https://github.com/s20522/Praca_in-ynierska_badania](https://github.com/s20522/Praca_in-ynierska_badania)

---

### Streszczenie

*(~200-250 słów) W tej sekcji należy zwięźle przedstawić całą pracę. Zacznij od wprowadzenia w problematykę niewydolności serca i podkreślenia znaczenia wczesnej predykcji. Następnie przedstaw cel pracy, czyli porównanie klasycznego modelu Random Forest z nowoczesnym podejściem opartym na sieciach neuronowych (MLP). Opisz krótko użytą metodologię: zbiór danych (299 pacjentów), przeprowadzone eksperymenty (EDA, budowa modelu bazowego, inżynieria cech, systematyczna optymalizacja MLP). Przedstaw kluczowe wyniki – wskaż, że model Random Forest okazał się lepszy (F1-score = 0.68, Recall = 89.5%) od najlepszej konfiguracji MLP (F1-score = 0.62). Zakończ głównym wnioskiem: w przypadku ograniczonego zbioru danych, dobrze dostrojony, klasyczny model może przewyższać bardziej złożone architektury deep learning, co podkreśla znaczenie doboru odpowiedniego narzędzia do skali problemu.*

---

### 1. Wstęp

*Ten rozdział ma za zadanie wprowadzić czytelnika w świat problemu, który rozwiązujesz. To jak zwiastun filmu – ma zaciekawić i pokazać, o co będzie chodzić, ale bez zdradzania wszystkich szczegółów.*

1.1. **Kontekst medyczny i znaczenie problemu**
    - **Co to jest niewydolność serca?** Wyjaśnij prostymi słowami, że to stan, w którym serce nie pompuje krwi tak wydajnie, jak powinno. To nie to samo co zawał! To przewlekła choroba.
    - **Dlaczego to ważne?** Podaj statystyki dotyczące rozpowszechnienia i śmiertelności (np. "jedna z głównych przyczyn hospitalizacji i zgonów na świecie"). Podkreśl, że wczesne zidentyfikowanie pacjentów wysokiego ryzyka pozwala lekarzom na wdrożenie intensywniejszego leczenia i potencjalne uratowanie życia.

1.2. **Problem badawczy i motywacja**
    - **Problem:** Jak na podstawie standardowych danych klinicznych przewidzieć, którzy pacjenci są najbardziej narażeni na zgon w okresie obserwacji?
    - **Motywacja:** Stworzenie narzędzia wspomagającego decyzje lekarskie. Wyobraź sobie system, który flaguje pacjenta jako "wysokie ryzyko", co skłania lekarza do dodatkowych badań lub częstszych wizyt kontrolnych.

1.3. **Cel i zakres pracy**
    - **Cel główny:** Porównanie skuteczności dwóch różnych filozofii modelowania – klasycznego, opartego na regułach modelu Random Forest oraz elastycznego, inspirowanego działaniem mózgu modelu sieci neuronowej (MLP) – w zadaniu predykcji śmiertelności pacjentów z niewydolnością serca.
    - **Pytania badawcze:**
        1.  Czy bardziej złożony model (MLP) jest w stanie osiągnąć lepsze wyniki niż prostszy (Random Forest) na tym konkretnym zbiorze danych?
        2.  Jakie techniki inżynierii cech (np. dyskretyzacja, interakcje) wpływają na wydajność modelu?
        3.  Jak poszczególne hiperparametry sieci neuronowej (architektura, regularyzacja) wpływają na jej zdolność predykcyjną?

1.4. **Struktura pracy**
    - Krótki przewodnik po kolejnych rozdziałach. Np. "W rozdziale drugim dokonano przeglądu literatury. Rozdział trzeci opisuje metodologię badawczą. W czwartym przedstawiono wyniki eksperymentów, które są następnie dyskutowane w rozdziale piątym, gdzie zaprezentowano również wnioski końcowe."

---

### 2. Przegląd Literatury (Literature Review)

*W tym rozdziale pokazujesz, że odrobiłeś pracę domową. Omawiasz, co już zostało zrobione w temacie i na jakich fundamentach teoretycznych opierasz swoją pracę. To buduje Twoją wiarygodność jako badacza.*

2.1. **Uczenie maszynowe w diagnostyce kardiologicznej**
    - Podaj przykłady, jak AI pomaga kardiologom. Np. automatyczna analiza obrazów EKG, segmentacja obrazów rezonansu magnetycznego serca, przewidywanie ryzyka zawału.

2.2. **Modele predykcyjne w niewydolności serca**
    - Skup się na badaniach podobnych do Twojego. Omów pracę Chicco & Jurman [1], która jest punktem wyjścia dla Twojej analizy. Wskaż, jakie modele testowali i jakie uzyskali wyniki. Możesz też wspomnieć o innych badaniach, które używały podobnych zbiorów danych.

2.3. **Podstawy teoretyczne wybranych algorytmów**
    - **Random Forest:**
        - **Jak to działa?** Wyjaśnij analogię "mądrości tłumu". Model buduje wiele prostych drzew decyzyjnych, a każde z nich "głosuje" na ostateczny wynik. To tak, jakbyś pytał o radę setki ekspertów, a potem wybierał najczęściej powtarzającą się odpowiedź.
        - **Źródła losowości:** Opisz dwa kluczowe mechanizmy: *bagging* (każde drzewo uczy się na nieco innej, losowej próbce danych) i *losowy wybór cech* (na każdym etapie budowy drzewa model losuje tylko podzbiór cech do analizy). To sprawia, że drzewa są zróżnicowane i model jest bardziej odporny.
        - **Fundamentalna publikacja:** Odwołaj się do przełomowej pracy Leo Breimana z 2001 roku [3], która formalnie wprowadziła ten algorytm.
    - **Sieci Neuronowe (MLP):**
        - **Inspiracja biologiczna:** Krótko o tym, że model jest luźno inspirowany budową mózgu (neurony, synapsy).
        - **Architektura:** Wyjaśnij pojęcia: warstwa wejściowa (Twoje dane), warstwy ukryte (gdzie dzieje się "magia"), warstwa wyjściowa (wynik). Pokaż prosty schemat.
        - **Proces uczenia (Backpropagation):** To serce sieci neuronowych. Wyjaśnij prostym językiem: sieć dokonuje predykcji, porównuje ją z prawdą, oblicza błąd, a następnie "propaguje" ten błąd wstecz, delikatnie korygując wagi połączeń, aby następnym razem pomylić się mniej. Odwołaj się do fundamentalnej pracy Rumelharta, Hintona i Williamsa z 1986 roku [4].

2.4. **Kluczowe komponenty sieci neuronowych**
    - **Funkcje Aktywacji:**
        - **Czym są?** To "przełączniki" w neuronach, które decydują, czy i jak silnie sygnał ma być przekazany dalej. Bez nich sieć neuronowa byłaby tylko prostym modelem liniowym.
        - **ReLU (Rectified Linear Unit):** Wyjaśnij, że to najpopularniejsza funkcja: `f(x) = max(0, x)`. Jest prosta i wydajna obliczeniowo. Odwołaj się do pracy Nair & Hinton z 2010 roku [5].
    - **Optymalizatory:**
        - **Czym są?** To algorytmy, które sterują procesem uczenia, decydując, jak duże kroki należy robić podczas korygowania wag. Wyobraź sobie schodzenie z góry we mgle – optymalizator pomaga znaleźć najszybszą drogę na sam dół (minimum błędu).
        - **Adam (Adaptive Moment Estimation):** Wyjaśnij, że to obecnie standardowy wybór. Jest "inteligentny" – dostosowuje tempo nauki dla każdego parametru osobno. Odwołaj się do pracy Kingma & Ba z 2014 roku [6].
    - **Techniki Regularyzacji:**
        - **Problem przeuczenia (Overfitting):** Wyjaśnij analogię do ucznia, który uczy się na pamięć, ale nie rozumie materiału. Model świetnie działa na danych treningowych, ale fatalnie na nowych.
        - **Dropout:** To genialna w swojej prostocie technika. Podczas każdej iteracji treningu losowo "wyłączamy" część neuronów. To zmusza sieć do budowania bardziej odpornej wiedzy, która nie zależy od pojedynczych neuronów. Odwołaj się do pracy Srivastava & Hinton z 2014 roku [7].

---

### 3. Metodologia i Środowisko Badawcze

*W tym rozdziale szczegółowo opisujesz, **CO** i **JAK** zrobiłeś. Musi być na tyle precyzyjny, aby inny badacz mógł powtórzyć Twoje eksperymenty krok po kroku.*

3.1. **Zbiór danych**
    - **Źródło i charakterystyka:** Opisz, skąd pochodzą dane (badanie Ahmada et al. [2]), ile zawierają próbek (299) i cech (13). Wymień najważniejsze cechy (wiek, frakcja wyrzutowa, kreatynina itp.).
    - **Zmienna celu:** Jasno zdefiniuj, co przewidujesz: `DEATH_EVENT` (1 - zgon, 0 - przeżycie).
    - **Tabela 1:** *Charakterystyka statystyczna zbioru danych.*
        - *(Wstaw tutaj tabelę wygenerowaną w EDA: `results/eda_fig_06_stats_table.png`)*
        - Opisz tabelę: wskaż średnie, odchylenia standardowe, wartości minimalne i maksymalne dla kluczowych cech.

3.2. **Eksploracyjna Analiza Danych (EDA)**
    - **Cel:** Wyjaśnij, że celem EDA było "zaprzyjaźnienie się" z danymi, zrozumienie ich struktury, znalezienie potencjalnych problemów i wstępnych zależności.
    - **Kroki:** Opisz, co zrobiłeś: analiza brakujących wartości (nie było), analiza rozkładu zmiennej celu, analiza korelacji, porównanie grup.
    - **Wykres 1:** *Rozkład zmiennej celu (`DEATH_EVENT`).*
        - *(Wstaw tutaj wykres słupkowy: `results/thesis_fig_01_target_distribution.png`)*
        - Opisz wykres: wskaż procentowy udział obu klas (68% vs 32%) i wyjaśnij pojęcie **niezbalansowania klas** (jedna klasa występuje znacznie częściej, co może być wyzwaniem dla modeli).
    - **Wykres 2:** *Porównanie rozkładów kluczowych cech w grupach pacjentów, którzy przeżyli i zmarli.*
        - *(Wstaw tutaj boxploty: `results/thesis_fig_03_key_features_comparison.png`)*
        - Opisz wykres: wskaż, że dla cech takich jak `ejection_fraction` czy `serum_creatinine` istnieją wyraźne, istotne statystycznie różnice między grupami.
    - **Problem "Target Leakage":** Wyjaśnij, dlaczego cecha `time` została wykluczona. To kluczowy element Twojej analizy! Użyj analogii: "Użycie tej cechy byłoby jak przewidywanie wyniku meczu, znając już czas jego zakończenia – drużyna, która przegrała, naturalnie grała krócej".

3.3. **Metryki oceny modeli**
    - Wyjaśnij, dlaczego sama trafność (accuracy) to za mało, zwłaszcza przy niezbalansowanych danych.
    - **Precision (Precyzja):** Jak wiele z naszych alarmów o "wysokim ryzyku" było trafnych? (TP / (TP + FP))
    - **Recall (Czułość):** Jak wiele osób z faktycznym wysokim ryzykiem udało nam się wykryć? (TP / (TP + FN)). Podkreśl, że w medycynie to często **najważniejsza metryka** – lepiej mieć fałszywy alarm niż kogoś przegapić.
    - **F1-score:** Średnia harmoniczna z Precision i Recall. Dobry, zbalansowany wskaźnik ogólnej skuteczności.
    - **AUC-ROC:** Miara zdolności modelu do odróżniania klas. Im bliżej 1.0, tym lepiej model separuje pacjentów zdrowych od chorych.

3.4. **Środowisko pracy**
    - Wymień narzędzia: Język programowania (Python 3.11), kluczowe biblioteki (scikit-learn, TensorFlow/Keras, pandas, matplotlib), środowisko (np. Jupyter Notebook, VS Code), system kontroli wersji (Git, GitHub).

---

### 4. Przeprowadzone Eksperymenty i Wyniki

*To najważniejszy rozdział Twojej pracy. Prezentujesz w nim wyniki swoich badań. Każdy podrozdział powinien mieć jasną strukturę: cel eksperymentu, opis metody, prezentacja wyników (tabela/wykres) i krótki wniosek.*

4.1. **Eksperyment 1: Budowa Modelu Bazowego (Random Forest)**
    - **Cel:** Stworzenie silnego, ale prostego modelu odniesienia.
    - **Metoda:** Opisz, jak zbudowałeś model: wybór 3 kluczowych cech (`age`, `ejection_fraction`, `serum_creatinine`), podział danych na zbiór treningowy i testowy (80/20 ze stratyfikacją), standaryzacja cech, optymalizacja hiperparametrów za pomocą `RandomizedSearchCV` z 5-krotną walidacją krzyżową.
    - **Wyniki:**
        - **Wykres 3:** *Macierz pomyłek dla zoptymalizowanego modelu Random Forest.*
            - *(Wstaw `results/rf_fig_01_confusion_matrix.png`)*
            - Opisz, co widać: ilu pacjentów zostało poprawnie zaklasyfikowanych, a ilu błędnie (błędy typu I i II).
        - **Tabela 2:** *Szczegółowe metryki wydajności dla modelu Random Forest.*
            - *(Wstaw `results/rf_fig_05_metrics_summary.png`)*
            - Podkreśl kluczowe wartości: F1-score (0.68) i fenomenalny Recall (89.5%).

4.2. **Eksperyment 2: Testowanie Inżynierii Cech**
    - **Cel:** Sprawdzenie, czy tworzenie nowych cech może poprawić wyniki modelu bazowego.
    - **Metoda:** Opisz testowane techniki:
        - **Dyskretyzacja:** Podział cech ciągłych na kategorie kliniczne.
        - **Cechy interakcyjne:** Tworzenie iloczynów cech (np. `age` * `serum_creatinine`).
    - **Wyniki:**
        - **Wykres 4:** *Porównanie metryki F1-score dla modelu bazowego i modeli z nowymi cechami.*
            - *(Wstaw `results/fe_fig_01_metrics_comparison.png`)*
            - Opisz wykres: pokaż, że żadna z technik nie przyniosła poprawy, a dyskretyzacja wręcz drastycznie pogorszyła wynik (spadek F1 o 25%).
    - **Wniosek:** Dla tego problemu, proste, surowe cechy okazały się najlepsze. To ważny wynik sam w sobie!

4.3. **Eksperyment 3: Systematyczna Budowa Sieci Neuronowej (MLP)**
    - **Cel:** Znalezienie optymalnej konfiguracji MLP poprzez serię kontrolowanych eksperymentów i porównanie jej z modelem bazowym.
    - **Metoda:** Opisz serię 5 mini-eksperymentów, w których testowałeś wpływ: architektury, funkcji aktywacji, regularyzacji Dropout, regularyzacji L2 i optymalizatorów.
    - **Wyniki:**
        - **Wykres 5:** *Porównanie kluczowych metryk dla najlepszych konfiguracji MLP oraz modelu Random Forest.*
            - *(Wstaw `results/nn_fig_06_mlp_vs_rf.png`)*
            - Wskaż na wykresie, że zielony słupek (Random Forest) dominuje, zwłaszcza w metryce Recall.
        - **Tabela 3:** *Ranking 10 najlepszych modeli ze wszystkich przeprowadzonych eksperymentów.*
            - *(Wstaw `results/nn_fig_07_summary_table.png`)*
            - Zwróć uwagę, że na szczycie tabeli znajduje się Random Forest, a najlepszy model MLP jest dopiero na drugim miejscu.
    - **Wniosek:** Mimo szeroko zakrojonej optymalizacji, żaden z 24 przetestowanych wariantów MLP nie zdołał pokonać prostszego modelu Random Forest.

---

### 5. Dyskusja i Wnioski

*To moment na refleksję. Nie tylko powtarzasz wyniki, ale interpretujesz je, zastanawiasz się, co oznaczają i jakie mają implikacje.*

5.1. **Dyskusja wyników**
    - **Dlaczego Random Forest wygrał?** Przedstaw argumenty: odporność na małe zbiory danych, mniejsza skłonność do przeuczania, skuteczność w znajdowaniu prostych reguł decyzyjnych, które w tym problemie okazały się wystarczające.
    - **Dlaczego MLP przegrał?** Argumenty: "głód danych" (data-hungry nature), wysoka wariancja i wrażliwość na hiperparametry, ryzyko przeuczenia na małym zbiorze (299 próbek to bardzo mało dla deep learningu).
    - **Znaczenie metryki Recall:** Jeszcze raz podkreśl, dlaczego wysoki Recall (89.5% w RF) jest tak cenny z perspektywy klinicznej. Lepiej ostrzec kilku pacjentów na wyrost, niż przegapić jednego, który faktycznie jest w grupie wysokiego ryzyka.

5.2. **Wnioski końcowe**
    - **Główny wniosek:** W niniejszej pracy udowodniono, że dla analizowanego zbioru danych, klasyczny model Random Forest jest skuteczniejszym narzędziem predykcyjnym niż bardziej złożone sieci neuronowe. Najlepszy model RF osiągnął F1-score na poziomie 0.68 i Recall 89.5%, znacząco przewyższając najlepszą konfigurację MLP (F1=0.62, Recall=73.7%).
    - **Wniosek metodologiczny:** Potwierdzono zasadę brzytwy Ockhama w uczeniu maszynowym – nie należy mnożyć bytów (złożoności modelu) ponad potrzebę. Prostsze modele często są nie tylko łatwiejsze w interpretacji, ale mogą być również skuteczniejsze, zwłaszcza przy ograniczonych danych.

5.3. **Ograniczenia pracy i kierunki dalszych badań**
    - **Ograniczenia:** Główne ograniczenie to **mały rozmiar zbioru danych**. Wyniki mogą nie być generalizowalne. Zbiór pochodzi z jednego ośrodka, co może wprowadzać pewne skrzywienie.
    - **Dalsze badania:**
        - Walidacja modeli na większym, wieloośrodkowym zbiorze danych.
        - Testowanie innych, bardziej zaawansowanych architektur sieci neuronowych, np. 1D-CNN na surowych szeregach czasowych (jeśli byłyby dostępne) lub modeli opartych o mechanizm uwagi (Transformers).
        - Zastosowanie zaawansowanych metod analizy przeżycia, takich jak DeepSurv, które modelują nie tylko fakt zgonu, ale i czas do zdarzenia.

---

### Bibliografia

[1] Chicco, D., & Jurman, G. (2020). Machine learning can predict survival of patients with heart failure from serum creatinine and ejection fraction alone. *BMC Medical Informatics and Decision Making*, 20(1), 16.

[2] Ahmad, T., Munir, A., Bhatti, S. H., Aftab, M., & Raza, M. A. (2017). Survival analysis of heart failure patients: a case study. *PloS one*, 12(7), e0181001.

[3] Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5-32.

[4] Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). Learning representations by back-propagating errors. *Nature*, 323(6088), 533-536.

[5] Nair, V., & Hinton, G. E. (2010). Rectified linear units improve restricted boltzmann machines. In *Proceedings of the 27th international conference on machine learning (ICML-10)* (pp. 807-814).

[6] Kingma, D. P., & Ba, J. (2014). Adam: A method for stochastic optimization. *arXiv preprint arXiv:1412.6980*.

[7] Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., & Salakhutdinov, R. (2014). Dropout: a simple way to prevent neural networks from overfitting. *The journal of machine learning research*, 15(1), 1929-1958.

---

### Spis Tabel i Wykresów

*Lista wszystkich tabel i wykresów użytych w pracy, z numerami i tytułami.*
