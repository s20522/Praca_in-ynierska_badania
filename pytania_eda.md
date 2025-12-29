# 🎓 Pytania do obrony pracy inżynierskiej - Eksploracyjna Analiza Danych (EDA)

**Autor:** Heart Failure Research Team  
**Data:** 29 grudnia 2024  
**Cel:** Przygotowanie do obrony pracy inżynierskiej poprzez zrozumienie kluczowych aspektów przeprowadzonej eksploracyjnej analizy danych.

---

## Wprowadzenie

Poniższa lista 30 pytań (wraz z odpowiedziami) została stworzona, aby pomóc w przygotowaniu się do obrony pracy inżynierskiej. Pytania obejmują zarówno podstawowe pojęcia, jak i szczegółową interpretację wyników uzyskanych w trakcie eksploracyjnej analizy danych. Celem jest nie tylko "wykucie" odpowiedzi, ale przede wszystkim **głębokie zrozumienie** materiału.

## Spis Kategorii

1.  [**I. Podstawy i Cel EDA (Pytania 1-5)**](#i-podstawy-i-cel-eda)
2.  [**II. Zbiór Danych i Zmienne (Pytania 6-10)**](#ii-zbiór-danych-i-zmienne)
3.  [**III. Metody i Techniki Analityczne (Pytania 11-20)**](#iii-metody-i-techniki-analityczne)
4.  [**IV. Interpretacja Wyników i Wnioski (Pytania 21-27)**](#iv-interpretacja-wyników-i-wnioski)
5.  [**V. Implikacje dla Modelowania (Pytania 28-30)**](#v-implikacje-dla-modelowania)

---

## I. Podstawy i Cel EDA

### Pytanie 1: Co to jest Eksploracyjna Analiza Danych (EDA) i jaki był jej główny cel w Pana/Pani pracy?

> **Odpowiedź:**
> Eksploracyjna Analiza Danych (EDA) to proces wstępnego badania danych w celu ich zrozumienia, podsumowania ich głównych cech, wykrycia wzorców i zidentyfikowania potencjalnych problemów. Głównym celem EDA w mojej pracy było **zrozumienie charakterystyki pacjentów z niewydolnością serca** oraz **zidentyfikowanie kluczowych czynników ryzyka** związanych ze śmiertelnością. Chciałem odpowiedzieć na pytanie: "Które cechy kliniczne najsilniej wpływają na przeżywalność pacjentów?", aby na tej podstawie zbudować skuteczny model predykcyjny.

### Pytanie 2: Dlaczego przeprowadzenie EDA jest ważne przed przystąpieniem do modelowania?

> **Odpowiedź:**
> EDA jest kluczowa, ponieważ bez niej modelowanie byłoby jak budowanie domu bez fundamentów. Po pierwsze, EDA pozwala **wykryć problemy w danych**, takie jak braki, błędy czy wartości odstające, które mogłyby zakłócić działanie modelu. Po drugie, pomaga **zrozumieć zależności** między zmiennymi, co jest niezbędne do wyboru odpowiednich cech i algorytmów. W mojej pracy dzięki EDA zidentyfikowałem problem **target leakage**, co uchroniło mnie przed popełnieniem poważnego błędu metodologicznego.

### Pytanie 3: Jakie były pierwsze kroki, które podjął/podjęła Pan/Pani w ramach EDA?

> **Odpowiedź:**
> Pierwszymi krokami było zapoznanie się ze "metryczką" zbioru danych. Sprawdziłem jego wymiary (299 pacjentów, 13 cech), typy danych w każdej kolumnie (czy są to liczby, czy kategorie) oraz zweryfikowałem, czy występują **brakujące wartości**. Upewnienie się, że zbiór jest kompletny i że typy danych są prawidłowe, było fundamentalne przed przejściem do bardziej zaawansowanych analiz.

### Pytanie 4: Jakie narzędzia/biblioteki zostały użyte do przeprowadzenia EDA?

> **Odpowiedź:**
> Analizę przeprowadziłem w języku Python, korzystając z następujących bibliotek:
> - **Pandas:** Do wczytywania, manipulacji i podstawowej analizy danych w strukturze DataFrame.
> - **Matplotlib i Seaborn:** Do tworzenia wizualizacji, takich jak histogramy, wykresy pudełkowe i macierze korelacji.
> - **NumPy:** Do operacji numerycznych.
> - **SciPy:** Do przeprowadzania testów statystycznych, takich jak test t-Studenta i test chi-kwadrat.

### Pytanie 5: Jakie główne problemy lub cechy charakterystyczne danych udało się zidentyfikować dzięki EDA?

> **Odpowiedź:**
> Dzięki EDA zidentyfikowałem trzy kluczowe cechy charakterystyczne:
> 1.  **Niezbalansowanie klas:** Stosunek pacjentów, którzy przeżyli, do tych, którzy zmarli, wynosił około 2:1, co wymaga specjalnego traktowania w modelowaniu.
> 2.  **Problem "target leakage":** Cecha `time` była silnie skorelowana ze zmienną celu, co jest błędem metodologicznym. Musiała zostać usunięta z modeli predykcyjnych.
> 3.  **Obecność wartości odstających:** W cechach takich jak kreatynina czy kinaza kreatynowa występowały wartości odstające, które jednak uznałem za klinicznie istotne i nie usuwałem ich.

---

## II. Zbiór Danych i Zmienne

### Pytanie 6: Proszę opisać zmienną celu w Pana/Pani projekcie.

> **Odpowiedź:**
> Zmienną celu w moim projekcie jest `DEATH_EVENT`. Jest to zmienna binarna, która przyjmuje wartość **1**, jeśli pacjent zmarł w okresie obserwacji, oraz **0**, jeśli przeżył. Analiza tej zmiennej wykazała, że 96 pacjentów (32.1%) zmarło, a 203 (67.9%) przeżyło, co wskazuje na niezbalansowanie klas.

### Pytanie 7: Co to jest "target leakage" i dlaczego cecha `time` stanowiła ten problem?

> **Odpowiedź:**
> **Target leakage** (przeciek informacji o celu) to sytuacja, w której dane używane do trenowania modelu zawierają informacje, które nie byłyby dostępne w momencie rzeczywistego przewidywania. Cecha `time` reprezentuje czas obserwacji pacjenta. Pacjenci, którzy zmarli, naturalnie mieli krótszy czas obserwacji. Użycie tej cechy w modelu byłoby oszustwem, ponieważ model nauczyłby się prostej zasady: "jeśli czas jest krótki, pacjent prawdopodobnie zmarł". W rzeczywistości nie znamy z góry czasu obserwacji, dlatego ta cecha musiała zostać usunięta z modeli predykcyjnych.

### Pytanie 8: Które cechy okazały się najważniejsze w kontekście przewidywania zgonu i dlaczego?

> **Odpowiedź:**
> Na podstawie analizy korelacji i porównania grup, cztery cechy okazały się najważniejsze:
> 1.  **Frakcja wyrzutowa:** Najsilniejsza ujemna korelacja (r=-0.269). Niższa wartość oznacza słabszą pracę serca i wyższe ryzyko.
> 2.  **Kreatynina w surowicy:** Najsilniejsza dodatnia korelacja (r=0.294). Wyższa wartość wskazuje na problemy z nerkami, co jest częstym powikłaniem.
> 3.  **Wiek:** Umiarkowana dodatnia korelacja (r=0.254). Starszy wiek jest naturalnym czynnikiem ryzyka.
> 4.  **Sód w surowicy:** Umiarkowana ujemna korelacja (r=-0.195). Niski poziom sodu (hiponatremia) jest znanym markerem złego rokowania.

### Pytanie 9: Czy w zbiorze danych występowały cechy kategoryczne? Jeśli tak, jak zostały przeanalizowane?

> **Odpowiedź:**
> Tak, w zbiorze występowało 5 cech binarnych (kategorycznych): płeć, palenie, cukrzyca, nadciśnienie i anemia. Zostały one przeanalizowane na dwa sposoby: po pierwsze, sprawdziłem ich rozkłady (np. ilu było palaczy, a ilu niepalących). Po drugie, użyłem **testu chi-kwadrat**, aby sprawdzić, czy istnieje istotny statystycznie związek między każdą z tych cech a zmienną celu (DEATH_EVENT). Analiza ta wykazała, że anemia i nadciśnienie mają istotny związek ze śmiertelnością.

### Pytanie 10: Co oznacza fakt, że w zbiorze danych nie było brakujących wartości?

> **Odpowiedź:**
> Oznacza to, że zbiór danych jest **kompletny** i wysokiej jakości. Brak brakujących wartości znacznie uprościł etap przygotowania danych, ponieważ nie musiałem stosować technik imputacji (czyli "wypełniania" braków, np. średnią lub medianą), które mogłyby wprowadzić dodatkowe szumy lub błędy do danych.

---

## III. Metody i Techniki Analityczne

### Pytanie 11: Co to jest korelacja i dlaczego jest ważna w Pana/Pani analizie?

> **Odpowiedź:**
> Korelacja to statystyczna miara, która określa siłę i kierunek związku między dwiema zmiennymi. W mojej analizie była kluczowa, ponieważ pozwoliła zidentyfikować, które cechy kliniczne są najsilniej powiązane ze zmienną celu (DEATH_EVENT). Na przykład, odkryłem silną ujemną korelację (r=-0.269) między frakcją wyrzutową a zgonem, co oznacza, że niższa frakcja wyrzutowa wiąże się z wyższym ryzykiem. To pomogło mi wybrać najważniejsze cechy do dalszego modelowania.

### Pytanie 12: Jak interpretować macierz korelacji (heatmapę)?

> **Odpowiedź:**
> Macierz korelacji w formie heatmapy to wizualna reprezentacja współczynników korelacji między wszystkimi parami zmiennych. Kolory pomagają szybko zidentyfikować silne związki: w mojej analizie, komórki bliskie czerwieni oznaczają silną korelację dodatnią, a bliskie niebieskiego - silną korelację ujemną. Wartości bliskie zera (białe) oznaczają brak korelacji. Analiza heatmapy pozwoliła mi również sprawdzić, czy nie ma problemu **multikolinearności**, czyli silnych korelacji między samymi cechami, co mogłoby zakłócić działanie niektórych modeli.

### Pytanie 13: Do czego służy test t-Studenta i co oznacza uzyskane p-value < 0.05?

> **Odpowiedź:**
> Test t-Studenta służy do porównywania średnich wartości w dwóch grupach. W mojej pracy użyłem go, aby sprawdzić, czy średnie wartości cech (np. wiek, kreatynina) różnią się w sposób **istotny statystycznie** między pacjentami, którzy przeżyli, a tymi, którzy zmarli. Uzyskane **p-value < 0.05** oznacza, że prawdopodobieństwo, iż zaobserwowana różnica jest dziełem przypadku, jest mniejsze niż 5%. Innymi słowy, możemy z dużym prawdopodobieństwem stwierdzić, że różnica jest rzeczywista i ma znaczenie.

### Pytanie 14: Czym jest test chi-kwadrat i w jakim celu został użyty?

> **Odpowiedź:**
> Test chi-kwadrat jest odpowiednikiem testu t-Studenta dla danych kategorycznych. Użyłem go, aby sprawdzić, czy istnieje istotny statystycznie związek między cechami binarnymi (np. anemia, nadciśnienie) a zmienną celu (DEATH_EVENT). Na przykład, test ten pokazał, że anemia występuje istotnie częściej w grupie pacjentów, którzy zmarli, niż w grupie, która przeżyła.

### Pytanie 15: Jakie wizualizacje były najbardziej pomocne w zrozumieniu danych i dlaczego?

> **Odpowiedź:**
> Najbardziej pomocne były dwie wizualizacje:
> 1.  **Wykresy pudełkowe (boxploty) porównujące grupy:** Pozwoliły one nie tylko zobaczyć różnice w średnich, ale także w rozproszeniu danych i zidentyfikować wartości odstające w obu grupach jednocześnie.
> 2.  **Wykres korelacji z DEATH_EVENT:** Uporządkowany wykres słupkowy, który w bardzo czytelny sposób pokazał, które cechy są najważniejsze i jaki jest kierunek ich wpływu (pozytywny czy negatywny).

### Pytanie 16: Co to są wartości odstające (outliers) i jak Pan/Pani je zidentyfikował/a?

> **Odpowiedź:**
> Wartości odstające to obserwacje, które znacznie odbiegają od pozostałych. Zidentyfikowałem je za pomocą **metody IQR (rozstępu międzykwartylowego)**. Wartość jest uznawana za odstającą, jeśli jest mniejsza niż pierwszy kwartyl minus 1.5 raza IQR lub większa niż trzeci kwartyl plus 1.5 raza IQR. Wizualnie można je zobaczyć na wykresach pudełkowych jako punkty poza "wąsami".

### Pytanie 17: Dlaczego nie usunął/usunęła Pan/Pani wartości odstających ze zbioru danych?

> **Odpowiedź:**
> Nie usunąłem wartości odstających, ponieważ w kontekście medycznym często niosą one **kluczową informację kliniczną**. Na przykład, bardzo wysoki poziom kreatyniny nie jest błędem pomiaru, lecz sygnałem ostrej niewydolności nerek, co jest stanem krytycznym. Usunięcie takich danych zubożyłoby model i sprawiło, że mógłby on gorzej radzić sobie z przewidywaniem najbardziej zagrożonych pacjentów.

### Pytanie 18: Co to jest rozkład danych i dlaczego analiza rozkładów (histogramy) jest ważna?

> **Odpowiedź:**
> Rozkład danych pokazuje, jak często występują różne wartości danej cechy. Analiza rozkładów za pomocą histogramów jest ważna, ponieważ pozwala zrozumieć charakterystykę zmiennej. Na przykład, analiza rozkładu kreatyniny pokazała, że jest on **prawostronnie skośny**, co oznacza, że większość pacjentów ma niskie wartości, ale istnieje "długi ogon" pacjentów z bardzo wysokimi wartościami. Ta wiedza jest ważna przy wyborze metod normalizacji danych.

### Pytanie 19: Czym różni się korelacja od przyczynowości?

> **Odpowiedź:**
> To bardzo ważna różnica. **Korelacja** oznacza jedynie, że dwie zmienne poruszają się razem (w tym samym lub przeciwnym kierunku). **Przyczynowość** oznacza, że zmiana jednej zmiennej **powoduje** zmianę w drugiej. EDA pozwala nam odkrywać korelacje, ale nie możemy na jej podstawie wnioskować o przyczynowości. Na przykład, stwierdziliśmy korelację między wiekiem a zgonem, ale to nie wiek sam w sobie jest przyczyną, lecz procesy starzenia się organizmu, które on reprezentuje.

### Pytanie 20: Jakie znaczenie ma niezbalansowanie klas dla analizy i modelowania?

> **Odpowiedź:**
> Niezbalansowanie klas (u nas 2:1) ma ogromne znaczenie. Jeśli byśmy je zignorowali, model mógłby osiągnąć wysoką dokładność (accuracy) po prostu przez przewidywanie zawsze klasy większościowej (czyli "przeżyje"). Taki model byłby bezużyteczny. Dlatego w dalszych etapach konieczne jest zastosowanie technik takich jak **oversampling** (np. SMOTE), **undersampling** lub **użycie wag klas**, aby model zwracał uwagę na obie klasy w równym stopniu. Należy też używać odpowiednich metryk, jak F1-score czy AUC-PR.

---

## IV. Interpretacja Wyników i Wnioski

### Pytanie 21: Jaki jest najważniejszy wniosek płynący z przeprowadzonej przez Pana/Panią EDA?

> **Odpowiedź:**
> Najważniejszy wniosek jest taki, że **przeżywalność pacjentów z niewydolnością serca jest silnie powiązana z mierzalnymi parametrami klinicznymi**, a w szczególności z funkcją serca (frakcja wyrzutowa) i funkcją nerek (kreatynina). Oznacza to, że istnieje solidna podstawa do zbudowania modelu predykcyjnego, który mógłby w przyszłości pomagać w identyfikacji pacjentów wysokiego ryzyka.

### Pytanie 22: Proszę zinterpretować ujemną korelację między frakcją wyrzutową a DEATH_EVENT.

> **Odpowiedź:**
> Ujemna korelacja (r=-0.269) oznacza, że im **niższa** jest frakcja wyrzutowa, tym **wyższe** jest ryzyko zgonu. Frakcja wyrzutowa to procent krwi wypompowywanej z lewej komory serca przy każdym skurczu. Niska wartość świadczy o słabej funkcji pompowania serca, co jest głównym objawem niewydolności serca. Nasza analiza potwierdziła, że jest to kluczowy wskaźnik prognostyczny.

### Pytanie 23: Dlaczego kreatynina w surowicy jest tak ważnym predyktorem?

> **Odpowiedź:**
> Kreatynina jest produktem przemiany materii wydalanym przez nerki. Jej wysoki poziom we krwi świadczy o tym, że nerki nie pracują prawidłowo. W niewydolności serca często dochodzi do tzw. **zespołu sercowo-nerkowego**, gdzie osłabione serce nie jest w stanie dostarczyć wystarczającej ilości krwi do nerek, co prowadzi do ich uszkodzenia. Dlatego wysoka kreatynina jest silnym sygnałem, że stan pacjenta jest poważny.

### Pytanie 24: Czy na podstawie EDA można stwierdzić, że palenie i cukrzyca nie mają wpływu na śmiertelność w tej grupie pacjentów?

> **Odpowiedź:**
> Nie, nie można tak stwierdzić. Nasza analiza (test chi-kwadrat) nie wykazała **istotnego statystycznie związku** między tymi cechami a zgonem w analizie jednowymiarowej. Nie oznacza to jednak, że nie mają one wpływu. Ich wpływ może być bardziej złożony, na przykład mogą być istotne w **interakcji z innymi cechami** (np. palenie może być szczególnie groźne u pacjentów w podeszłym wieku). Dopiero modele wielowymiarowe (multivariate) mogą w pełni ocenić ich znaczenie.

### Pytanie 25: Jakie grupy pacjentów, na podstawie analizy, wydają się być najbardziej narażone na ryzyko zgonu?

> **Odpowiedź:**
> Na podstawie analizy, najbardziej narażeni wydają się być pacjenci, którzy charakteryzują się kombinacją kilku czynników ryzyka. Analiza wykresów punktowych (scatter plot) sugeruje, że grupa najwyższego ryzyka to **starsi pacjenci (wiek > 60 lat) z niską frakcją wyrzutową (<30%) oraz podwyższonym poziomem kreatyniny (>1.2 mg/dL) i obniżonym poziomem sodu (<135 mEq/L)**.

### Pytanie 26: Co mówi nam fakt, że cechy takie jak wiek i frakcja wyrzutowa są ze sobą ujemnie skorelowane (r=-0.439)?

> **Odpowiedź:**
> Oznacza to, że istnieje tendencja, że **starsi pacjenci mają niższą frakcję wyrzutową**. Jest to zgodne z wiedzą medyczną, ponieważ z wiekiem funkcja serca naturalnie się pogarsza. Ta korelacja jest umiarkowana, co oznacza, że wiek nie jest jedynym czynnikiem determinującym frakcję wyrzutową, ale ma na nią istotny wpływ.

### Pytanie 27: Czy wyniki Pana/Pani EDA są zgodne z ogólną wiedzą medyczną na temat niewydolności serca?

> **Odpowiedź:**
> Tak, wyniki są w pełni zgodne z wiedzą medyczną. Zidentyfikowane kluczowe czynniki ryzyka - niska frakcja wyrzutowa, wysoka kreatynina, podeszły wiek i hiponatremia - są powszechnie uznawane w kardiologii za najważniejsze wskaźniki prognostyczne w niewydolności serca. To potwierdza, że nasz zbiór danych jest reprezentatywny i że analiza zmierza w dobrym kierunku.

---

## V. Implikacje dla Modelowania

### Pytanie 28: W jaki sposób wyniki EDA wpłyną na etap przygotowania danych (preprocessing) przed modelowaniem?

> **Odpowiedź:**
> Wyniki EDA mają bezpośredni wpływ na preprocessing:
> 1.  **Wykluczenie cechy `time`:** Ze względu na target leakage.
> 2.  **Normalizacja/Standaryzacja:** Cechy numeryczne mają różne skale (np. wiek 40-95, a kreatynina 0.5-9.4). Konieczne będzie ich przeskalowanie (np. za pomocą StandardScaler), aby modele takie jak sieci neuronowe działały poprawnie.
> 3.  **Obsługa wartości odstających:** Zamiast standardowej normalizacji, można rozważyć `RobustScaler`, który jest mniej wrażliwy na outliery.
> 4.  **Balansowanie klas:** Należy zastosować techniki takie jak SMOTE lub wagi klas, aby model nie faworyzował klasy większościowej.

### Pytanie 29: Czy na podstawie EDA można zaproponować jakieś nowe cechy (feature engineering)?

> **Odpowiedź:**
> Tak, EDA inspiruje do stworzenia kilku nowych cech:
> 1.  **Kategorie wiekowe:** Zamiast ciągłej cechy `age`, można stworzyć kategorie (np. "<60", "60-75", ">75"), co może pomóc modelom drzewiastym.
> 2.  **Wskaźnik sercowo-nerkowy:** Można stworzyć cechę będącą kombinacją frakcji wyrzutowej i kreatyniny (np. ich iloraz lub iloczyn), aby uchwycić ich połączony, negatywny efekt.
> 3.  **Flagi kliniczne:** Można stworzyć binarne flagi dla wartości przekraczających progi kliniczne (np. `is_ef_low` dla frakcji wyrzutowej < 30%, `is_creatinine_high` dla kreatyniny > 1.2 mg/dL).

### Pytanie 30: Jakie typy modeli uczenia maszynowego wydają się być najbardziej obiecujące na podstawie wyników EDA?

> **Odpowiedź:**
> Wyniki EDA sugerują, że warto przetestować kilka typów modeli:
> 1.  **Modele drzewiaste (Random Forest, XGBoost, LightGBM):** Są one odporne na wartości odstające i dobrze radzą sobie z nieliniowymi zależnościami, które mogą występować w danych.
> 2.  **Maszyny wektorów nośnych (SVM):** Mogą być skuteczne po odpowiedniej normalizacji danych, zwłaszcza z jądrem nieliniowym (np. RBF).
> 3.  **Sieci neuronowe (MLP):** Mają potencjał do odkrywania bardzo złożonych wzorców w danych, ale wymagają starannego preprocessingu (normalizacja, balansowanie klas) i optymalizacji hiperparametrów.

EDA pokazała, że problem jest złożony i nie ma jednej, oczywistej zależności, dlatego porównanie różnych podejść będzie kluczowe.
