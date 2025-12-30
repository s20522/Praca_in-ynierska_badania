# 🎓 Pytania do obrony pracy inżynierskiej - Inżynieria Cech (Feature Engineering)

**Autor:** Heart Failure Research Team  
**Data:** 29 grudnia 2024  
**Cel:** Przygotowanie do obrony pracy inżynierskiej poprzez zrozumienie przeprowadzonych eksperymentów z inżynierią cech, ich wyników oraz płynących z nich wniosków.

---

## Wprowadzenie

Poniższa lista 30 pytań (wraz z odpowiedziami) została stworzona, aby pomóc w przygotowaniu się do obrony pracy inżynierskiej w sekcji dotyczącej **inżynierii cech**. Pytania te mają na celu nie tylko sprawdzenie wiedzy, ale przede wszystkim ocenę zdolności do **krytycznej analizy wyników**, wyciągania wniosków i uzasadniania podjętych decyzji projektowych. To kluczowa część oryginalnego wkładu badawczego.

## Spis Kategorii

1.  [**I. Podstawy i Motywacje (Pytania 1-6)**](#i-podstawy-i-motywacje)
2.  [**II. Eksperyment z Dyskretyzacją (Pytania 7-11)**](#ii-eksperyment-z-dyskretyzacją)
3.  [**III. Eksperyment z Cechami Interakcyjnymi (Pytania 12-17)**](#iii-eksperyment-z-cechami-interakcyjnymi)
4.  [**IV. Eksperymenty z Normalizacją i Wszystkimi Cechami (Pytania 18-22)**](#iv-eksperymenty-z-normalizacją-i-wszystkimi-cechami)
5.  [**V. Podsumowanie i Wnioski Ogólne (Pytania 23-30)**](#v-podsumowanie-i-wnioski-ogólne)

---

## I. Podstawy i Motywacje

### Pytanie 1: Proszę wyjaśnić, czym jest inżynieria cech i dlaczego jest to ważny etap w procesie budowy modeli uczenia maszynowego?

> **Odpowiedź:**
> Inżynieria cech to proces tworzenia nowych, bardziej informatywnych cech (zmiennych) z istniejących danych surowych. Jest to kluczowy etap, ponieważ jakość i forma danych wejściowych mają bezpośredni wpływ na skuteczność modelu. Celem jest takie przekształcenie danych, aby lepiej reprezentowały strukturę problemu i ułatwiły modelowi "naukę" kluczowych wzorców. Dobrze przeprowadzona inżynieria cech może znacząco poprawić wyniki, podczas gdy słabe cechy mogą ograniczyć nawet najlepszy algorytm.

### Pytanie 2: Jaki był cel przeprowadzania eksperymentów z inżynierią cech w Pana/Pani pracy?

> **Odpowiedź:**
> Celem było zbadanie, czy poprzez systematyczne tworzenie nowych cech można poprawić skuteczność predykcyjną modelu bazowego Random Forest. Chciałem/am sprawdzić, czy techniki takie jak dyskretyzacja, tworzenie interakcji czy użycie większej liczby cech pozwolą modelowi lepiej uchwycić złożone zależności w danych i w konsekwencji osiągnąć wyższe wartości metryk, zwłaszcza F1-score i Recall.

### Pytanie 3: Co to jest model bazowy (baseline) i dlaczego jego ustanowienie było kluczowe przed rozpoczęciem eksperymentów?

> **Odpowiedź:**
> Model bazowy to prosty, ale sensowny model, który służy jako **punkt odniesienia** dla wszystkich dalszych eksperymentów. W moim przypadku był to model Random Forest wytrenowany na trzech najważniejszych surowych cechach. Ustanowienie baseline jest kluczowe, ponieważ bez niego nie bylibyśmy w stanie obiektywnie ocenić, czy nasze bardziej skomplikowane podejścia (np. z inżynierią cech) faktycznie przynoszą poprawę, czy może wręcz pogarszają wyniki.

### Pytanie 4: Jakie trzy główne kierunki inżynierii cech zostały zbadane w Pana/Pani pracy?

> **Odpowiedź:**
> Zbadane zostały trzy główne kierunki:
> 1.  **Dyskretyzacja:** Zamiana cech ciągłych (wiek, kreatynina, frakcja wyrzutowa) na kategorie oparte na progach klinicznych.
> 2.  **Tworzenie cech interakcyjnych:** Generowanie nowych cech poprzez mnożenie istniejących (np. wiek × kreatynina), aby uchwycić efekty synergii.
> 3.  **Zmiana liczby i skali cech:** Testowanie wpływu użycia wszystkich dostępnych cech oraz alternatywnej metody normalizacji (MinMaxScaler).

### Pytanie 5: Jak zapewniono spójność i porównywalność wyników między poszczególnymi eksperymentami?

> **Odpowiedź:**
> Aby zapewnić porównywalność, wszystkie eksperymenty były prowadzone w kontrolowanych warunkach. Użyto tego samego podziału na zbiór treningowy i testowy (`random_state=42`), tej samej strategii walidacji krzyżowej oraz tych samych, zoptymalizowanych wcześniej, hiperparametrów dla modelu Random Forest. Dzięki temu wszelkie różnice w wynikach można było z dużą pewnością przypisać wyłącznie wpływowi zastosowanej techniki inżynierii cech.

### Pytanie 6: Czy spodziewał/a się Pan/Pani, że inżynieria cech poprawi wyniki? Jakie były Pana/Pani wstępne hipotezy?

> **Odpowiedź:**
> Moją wstępną hipotezą było, że inżynieria cech, a zwłaszcza tworzenie cech interakcyjnych, powinna poprawić wyniki. Spodziewałem/am się, że interakcje, takie jak wiek × kreatynina, lepiej oddadzą złożoność procesów fizjologicznych i dostarczą modelowi cenniejszych informacji niż pojedyncze cechy. Byłem/am bardziej sceptyczny/a co do dyskretyzacji, ale chciałem/am to sprawdzić empirycznie. Wyniki okazały się zaskakujące i obaliły moją główną hipotezę.

---

## II. Eksperyment z Dyskretyzacją

### Pytanie 7: Na czym polegała dyskretyzacja i jakie było jej uzasadnienie teoretyczne?

> **Odpowiedź:**
> Dyskretyzacja polegała na zamianie wartości ciągłych na kategorie. Na przykład, wiek został podzielony na przedziały [40-60], [60-80] itd. Teoretyczne uzasadnienie było takie, że modele drzewiaste (jak Random Forest) naturalnie operują na progach, a podanie im gotowych, sensownych klinicznie kategorii mogłoby ułatwić im zadanie. Ponadto, klinicyści często myślą w kategoriach przedziałów referencyjnych, więc taki model mógłby być bardziej interpretowalny.

### Pytanie 8: Jakie były wyniki modelu po zastosowaniu dyskretyzacji?

> **Odpowiedź:**
> Wyniki były **bardzo słabe**. F1-score spadł z 0.6800 do 0.5106, co stanowi spadek o 25%. Podobnie, Recall spadł z 89.5% do 63.2%. Była to zdecydowanie najgorsza ze wszystkich testowanych strategii.

### Pytanie 9: Dlaczego, Pana/Pani zdaniem, dyskretyzacja tak drastycznie pogorszyła wyniki?

> **Odpowiedź:**
> Główną przyczyną była **znacząca utrata informacji**. Zamieniając dokładną wartość, np. kreatyniny 1.25 mg/dL, na ogólną kategorię "podwyższony", pozbawiamy model precyzyjnej informacji. Dla modelu nie ma już różnicy między wartością 1.25 a 2.9, mimo że klinicznie jest ona ogromna. Random Forest sam potrafi znaleźć optymalne progi podziału, a narzucenie mu z góry zdefiniowanych, szerokich kategorii tylko mu w tym przeszkodziło.

### Pytanie 10: Czy istnieją sytuacje, w których dyskretyzacja mogłaby być korzystna?

> **Odpowiedź:**
> Tak, dyskretyzacja może być korzystna w pewnych sytuacjach. Na przykład, gdy mamy do czynienia z algorytmami, które słabo radzą sobie z nieliniowościami (np. regresja logistyczna), lub gdy dane są bardzo zaszumione i precyzyjne wartości nie są wiarygodne. Może też pomóc, gdy mamy bardzo mało danych i chcemy uniknąć przeuczenia. Jednak w przypadku nowoczesnych, nieliniowych modeli jak Random Forest, jest to często niepotrzebne, a nawet szkodliwe.

### Pytanie 11: Jakie progi zostały użyte do dyskretyzacji i na jakiej podstawie zostały wybrane?

> **Odpowiedź:**
> Progi zostały wybrane na podstawie **powszechnie przyjętych norm klinicznych**. Na przykład, dla kreatyniny próg 1.2 mg/dL jest często granicą normy, a dla frakcji wyrzutowej próg 30% oddziela ciężką dysfunkcję od umiarkowanej. Celem było stworzenie kategorii, które byłyby zrozumiałe i sensowne z medycznego punktu widzenia.

---

## III. Eksperyment z Cechami Interakcyjnymi

### Pytanie 12: Co to są cechy interakcyjne i jaki efekt synergii miały reprezentować w Pana/Pani projekcie?

> **Odpowiedź:**
> Cechy interakcyjne powstają przez połączenie (np. pomnożenie) dwóch lub więcej cech. Mają one na celu uchwycenie **efektu synergii**, gdzie połączony wpływ dwóch czynników jest inny niż suma ich pojedynczych wpływów. Na przykład, cecha `age_x_creat` miała reprezentować skumulowane ryzyko, gdzie wysoki poziom kreatyniny jest znacznie groźniejszy u osoby starszej niż u młodej.

### Pytanie 13: Jakie były wyniki modelu po dodaniu cech interakcyjnych?

> **Odpowiedź:**
> Wyniki były **nieco gorsze** niż w modelu bazowym. F1-score spadł nieznacznie z 0.6800 do 0.6667, a Recall z 89.5% do 84.2%. Mimo że spadek nie był duży, żadna z metryk nie uległa poprawie.

### Pytanie 14: Analiza ważności cech pokazała, że nowe cechy interakcyjne zdominowały oryginalne. Jak to możliwe, że mimo to ogólny wynik modelu się pogorszył?

> **Odpowiedź:**
> To bardzo ciekawa obserwacja. Fakt, że cecha `age_x_creat` uzyskała aż 43.6% ważności, świadczy o tym, że model uznał ją za bardzo predykcyjną. Prawdopodobnie ta jedna, połączona cecha niosła w sobie większość informacji z `age` i `serum_creatinine`. Pogorszenie wyniku mogło wynikać z kilku przyczyn: po pierwsze, dodanie nowych cech mogło wprowadzić dodatkowy **szum lub redundancję**, która utrudniła optymalizację. Po drugie, możliwe, że model stał się **zbyt skoncentrowany** na tych nowych, silnych cechach, ignorując subtelniejsze sygnały z innych zmiennych.

### Pytanie 15: Czy uważa Pan/Pani, że inny sposób tworzenia interakcji (np. dzielenie, a nie mnożenie) mógłby dać lepsze rezultaty?

> **Odpowiedź:**
> Jest to możliwe. Mnożenie jest najprostszą formą tworzenia interakcji. Można by zbadać inne, bardziej złożone transformacje, na przykład dzielenie (np. tworząc wskaźnik `serum_creatinine / ejection_fraction`) lub użycie transformacji wielomianowych. Każda z tych metod mogłaby uchwycić inny rodzaj zależności. Jednak w ramach tej pracy skupiłem/am się na najpopularniejszej i najbardziej intuicyjnej metodzie, jaką jest mnożenie.

### Pytanie 16: Dlaczego do jednej z interakcji użyto cechy `serum_sodium`, która nie była w modelu bazowym?

> **Odpowiedź:**
> Zostało to zrobione celowo, aby przetestować hipotezę, że interakcja dwóch cech, które indywidualnie nie są najsilniejszymi predyktorami, może stworzyć silny, nowy sygnał. W analizie EDA `serum_sodium` wykazywało pewną korelację z wynikiem. Interakcja `ef_x_sodium` miała reprezentować złożony stan pacjenta, łącząc dysfunkcję serca z zaburzeniami elektrolitowymi. Jak pokazały wyniki, ta nowa cecha uzyskała aż 29% ważności, co potwierdza, że była to ciekawa i informatywna kombinacja.

### Pytanie 17: Czy cechy interakcyjne są trudniejsze do interpretacji?

> **Odpowiedź:**
> Tak, mogą być trudniejsze. Interpretacja pojedynczej cechy jak "wiek" jest prosta. Interpretacja cechy "wiek × kreatynina" jest bardziej złożona i wymaga myślenia o połączonym efekcie. Jednak w kontekście medycznym takie interakcje są często bardziej naturalne i intuicyjne dla lekarzy, którzy wiedzą, że czynniki ryzyka rzadko działają w izolacji. Zatem, mimo że są matematycznie bardziej złożone, mogą prowadzić do bardziej realistycznych i użytecznych klinicznie wniosków.

---

## IV. Eksperymenty z Normalizacją i Wszystkimi Cechami

### Pytanie 18: Dlaczego przetestowano `MinMaxScaler` jako alternatywę dla `StandardScaler`?

> **Odpowiedź:**
> Zostało to zrobione w celu **weryfikacji teoretycznego założenia**, że modele drzewiaste, takie jak Random Forest, są niewrażliwe na skalę cech. `StandardScaler` i `MinMaxScaler` to dwie najpopularniejsze techniki skalowania, które działają w różny sposób. Chciałem/am empirycznie potwierdzić, czy zmiana tej techniki będzie miała jakikolwiek wpływ na wyniki. Eksperyment pokazał, że nie miała żadnego, co jest cennym potwierdzeniem właściwości algorytmu.

### Pytanie 19: Jakie były wyniki po zastosowaniu `MinMaxScaler`?

> **Odpowiedź:**
> Wyniki były **identyczne** z modelem bazowym. F1-score, Recall, Precision i AUC-ROC nie zmieniły się nawet o ułamek procenta. To jednoznacznie dowodzi, że dla Random Forest metoda skalowania danych nie ma znaczenia.

### Pytanie 20: Jaka była motywacja stojąca za eksperymentem z użyciem wszystkich dostępnych cech?

> **Odpowiedź:**
> Motywacją było sprawdzenie hipotezy, czy model, mając do dyspozycji **pełen obraz kliniczny pacjenta** (14 cech), będzie w stanie znaleźć bardziej złożone i subtelne wzorce, które zostały pominięte w modelu z tylko 3 cechami. Była to próba sprawdzenia, czy podejście "więcej informacji = lepsze wyniki" sprawdzi się w tym przypadku.

### Pytanie 21: Jakie były wyniki modelu wytrenowanego na wszystkich cechach i jak Pan/Pani je interpretuje?

> **Odpowiedź:**
> Wyniki były **gorsze** niż w modelu bazowym. F1-score spadł o 4%, a AUC-ROC o prawie 6%. Interpretuję to jako klasyczny przykład **"klątwy wymiarowości"**. Dodanie wielu mniej istotnych cech wprowadziło do modelu "szum", który utrudnił mu skupienie się na najważniejszych sygnałach. Model zaczął się uczyć przypadkowych fluktuacji zamiast rzeczywistych zależności, co pogorszyło jego zdolność generalizacji.

### Pytanie 22: Czy analiza ważności cech w modelu z 14 cechami przyniosła jakieś nowe, interesujące wnioski?

> **Odpowiedź:**
> Tak. Po pierwsze, potwierdziła dominację cech interakcyjnych, które stworzyliśmy (`age_x_creat` i `ef_x_sodium` były na szczycie). Po drugie, pokazała, że cechy takie jak `platelets` (płytki krwi) i `creatinine_phosphokinase` (CPK), które nie były w modelu bazowym, mają pewną, choć niewielką, moc predykcyjną. To sugeruje, że mogłyby być one kandydatami do dalszych, bardziej zaawansowanych modeli, ale w prostym podejściu więcej szkodziły niż pomagały.

---

## V. Podsumowanie i Wnioski Ogólne

### Pytanie 23: Jaki jest najważniejszy, ogólny wniosek płynący z przeprowadzonych eksperymentów z inżynierią cech?

> **Odpowiedź:**
> Najważniejszy wniosek jest taki, że w przypadku tego konkretnego problemu i zbioru danych, **najprostszą i najskuteczniejszą strategią okazała się staranna selekcja kilku najważniejszych, surowych cech**. Żadna z testowanych, bardziej zaawansowanych technik inżynierii cech nie przyniosła poprawy. To pokazuje, że kluczem do sukcesu jest często **jakość i trafność cech, a nie ich ilość czy złożoność**.

### Pytanie 24: Czy uważa Pan/Pani, że te eksperymenty były porażką, skoro nie udało się poprawić modelu bazowego?

> **Odpowiedź:**
> Absolutnie nie. W badaniach naukowych **wynik negatywny jest również cennym wynikiem**. Te eksperymenty nie były porażką, ale **sukcesem w weryfikacji hipotez**. Udowodniliśmy empirycznie, że pewne popularne techniki w tym przypadku nie działają, co jest bardzo ważną informacją. Potwierdziliśmy również optymalność naszego modelu bazowego, co wzmacnia wiarygodność całej pracy. Pokazanie, co nie działa, jest równie ważne, jak pokazanie, co działa.

### Pytanie 25: Jakie znaczenie dla praktyki uczenia maszynowego ma wniosek "mniej znaczy więcej"?

> **Odpowiedź:**
> Ma to ogromne znaczenie. Oznacza, że zamiast bezrefleksyjnie "wrzucać" do modelu wszystkie dostępne dane, należy poświęcić czas na **zrozumienie problemu i danych (EDA)**, aby zidentyfikować kluczowe predyktory. Prostsze modele, oparte na mniejszej liczbie cech, są nie tylko często równie (lub bardziej) skuteczne, ale także **szybsze w treningu, łatwiejsze do wdrożenia i, co najważniejsze, znacznie łatwiejsze do interpretacji** i zrozumienia, co jest kluczowe w zastosowaniach krytycznych, takich jak medycyna.

### Pytanie 26: Proszę spojrzeć na wykres porównujący F1-score i Recall. Który model, oprócz bazowego, uznałby Pan/Pani za "drugi najlepszy" i dlaczego?

> **Odpowiedź:**
> Za "drugi najlepszy" uznałbym/abym model z **cechami interakcyjnymi**. Mimo że jego F1-score był nieco niższy od bazowego, wciąż utrzymywał bardzo wysoki Recall (84.2%), co jest dla nas priorytetem. W przeciwieństwie do modelu z dyskretyzacją, nie stracił on drastycznie zdolności do wykrywania pacjentów wysokiego ryzyka. Spadek wydajności był na tyle mały, że w innym scenariuszu mógłby być akceptowalnym kompromisem za potencjalnie ciekawsze wnioski płynące z analizy interakcji.

### Pytanie 27: Jakie ograniczenia miały przeprowadzone eksperymenty?

> **Odpowiedź:**
> Główne ograniczenia to:
> - **Ograniczony zakres testowanych technik:** Skupiłem/am się na kilku popularnych metodach. Istnieje wiele innych, bardziej zaawansowanych technik inżynierii cech, których nie zbadałem/am.
> - **Jeden algorytm:** Wszystkie eksperymenty były prowadzone na modelu Random Forest. Możliwe, że cechy, które nie pomogły temu modelowi (np. interakcje), mogłyby być bardziej użyteczne dla innego typu algorytmu, np. sieci neuronowej.
> - **Mały zbiór danych:** Na większym zbiorze danych wyniki mogłyby być inne, a subtelniejsze efekty mogłyby stać się bardziej widoczne.

### Pytanie 28: Jakie wnioski z tych eksperymentów weźmie Pan/Pani pod uwagę w kolejnych etapach pracy, np. przy budowie sieci neuronowych?

> **Odpowiedź:**
> Przede wszystkim, wiem już, że **selekcja cech jest kluczowa**. Rozpoczynając pracę z sieciami neuronowymi, na pewno zacznę od prostego modelu opartego na tych samych trzech, najsilniejszych cechach. Będę również bardzo ostrożny/a z dodawaniem dużej liczby cech, wiedząc, że może to wprowadzić szum. Ponadto, wiem, że skalowanie danych jest absolutnie konieczne dla sieci neuronowych, a wyniki eksperymentu z `MinMaxScaler` i `StandardScaler` mogą pomóc w wyborze odpowiedniej strategii.

### Pytanie 29: Czy uważa Pan/Pani, że wyniki byłyby inne, gdyby użyto innego algorytmu niż Random Forest?

> **Odpowiedź:**
> Jest to bardzo prawdopodobne. Różne algorytmy "patrzą" na dane w różny sposób. Na przykład, modele liniowe (jak regresja logistyczna) mogłyby bardziej skorzystać na ręcznie stworzonych interakcjach, ponieważ same nie potrafią ich modelować. Z kolei sieci neuronowe, dzięki swojej złożonej architekturze, potrafią same uczyć się bardzo skomplikowanych, nieliniowych interakcji, więc dodawanie ich ręcznie może być mniej potrzebne. Dlatego ważne jest testowanie różnych kombinacji cech i modeli.

### Pytanie 30: Podsumowując, jaka jest Pana/Pani ostateczna rekomendacja dotycząca inżynierii cech dla tego problemu?

> **Odpowiedź:**
> Moja ostateczna rekomendacja, oparta na przeprowadzonych eksperymentach, jest następująca: dla tego zbioru danych i celu predykcyjnego, **najlepszą strategią jest skupienie się na prostym modelu wykorzystującym trzy kluczowe, surowe predyktory: wiek, frakcję wyrzutową i poziom kreatyniny w surowicy**. Czas i wysiłek, które można by poświęcić na złożoną inżynierię cech, lepiej zainwestować w staranną optymalizację hiperparametrów wybranego modelu lub w testowanie różnych architektur algorytmów na tym sprawdzonym, wysokiej jakości zestawie cech.
