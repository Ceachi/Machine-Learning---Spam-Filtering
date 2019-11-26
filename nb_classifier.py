from collections import Counter, defaultdict
import numpy as np
from pprint import pprint

'''
Native Bayes (implementarea algoritmului in Python)
https://www.geeksforgeeks.org/naive-bayes-classifiers/

In primul rand avem nevoie de datele de antrenare, de exemplu:

Index simplu | Temperatura(x1) | Vreme(x2) | Umiditate(x3) | Plec in vacanta(Yj)?
    0               Rece          Inorat        Mare                NU
    1               Placut        Senin         Normal              DA
    2               Cald          Insorit       Mare                NU
    3               Cald          Insorit       Mica                DA


1) Trebuie sa ne dam seama cate clase sunt si care sunt variatile pt fiecare atribut? 
    R: sunt doua clase Y1('Da') si Y2('Nu')
        X1 -> Rece, Placut, Cald
        X2 -> Inorat, Senin, Insorit
        X3-> Mare, Normal, Mic

2) Se calculeaza probabilitatea de aparitie a fiecarei clase
        P(Y1) = P('DA') = 2/4 = 1/2 = 50%
        P(Y2) = P('NU') = 2/4 = 1/2 = 50%

3) Se calculeaza probabilitatile de aparitie pentru fiecare feature(atribut)
        P(Rece) = 1/4 = 25%
        P(Cald) = 2/4 = 50%
        P(Inorat) = 1/4 = 25%
        P(Normal) = 1/4 = 25%
        ..... si tot asa pt toate

4) Se calculeaza probabilitatile conditionate P(X|Y)
    P(Rece|NU) = 1/2 = 50% (avem 2 pentru clasa "nu" si Rece apare 1 singura data)
    P(Rece|Da) = 0% (Rece nu apare nicaieri in clasa "Da")
    P(Senin|Nu) = 0%
    P(senint|Da) = 50%

(pana in punctul asta ai realizat procesul de antrenare
'''


class NBClassifier():
    ''' Metoda pentru antrenarea modelului

    - aici populam dictionarul in care sctocam informatiile din antrenare(self.nb_dict) si tot aici
    se vor defini cateva atribute ale obiectului ce vor fi ultile in procesul de antrenare.

    X = numpy-array, contine feature-urile pentru exemplele de antrenare
    Y = este o lista ce contine label-urile/etichetele pentru exemplele de antrenare

    '''
    def train(self, X, y):
        self.data = X
        self.labels = np.unique(y) #clasele gasite in dataset si le luam pe cele unice
        self.class_probabilities = self._calculate_relative_proba(y) #stocheaza prob de aparitie a fiecarei clase { 'Da' = 0.5, 'NU' = 0.5 }
        pprint(self.class_probabilities)
        self._initialize_nb_dict()  # Creates the dict which stores the information gathered durin training
        examples_no, features_no = self.data.shape

        '''
        Pentru fiecare clasa prezentata in dataset, o sa cream o lista X_class ce o sa contina exemplele de antrenare 
        pentru clasa respectiva.
        
        De exemplu: X_Class = [ ['Placut', 'Senin", 'Normala'], ['Cald', 'Insorit', 'Mica']] --> Pentru clasele 'Da'
        X_Class = X = [    ['Rece', 'Innorat', 'Mare'], ['Cald', Insorit, 'Mare']] ---> Pentru clasele 'Nu'
        
        Mai departe, o să creăm o cheie pentru fiecare feature (x1, x2, ..., xn) în dicționarele ce reprezintă valori 
        pentru cheile inițiale (”DA” și ”NU”). 
        La aceste chei ce reprezintă tipul de feature, o să adaugăm în listă toate aparițiile pentru tipul respectiv
         de feature ținând cont de clasa
        La finalul acestui bloc de cod, self.nb_dict va avea forma:{

        'DA': defaultdict(<class 'list'>,
        
                           {0: ['Placut', 'Cald'],
        
                            1: ['Senin', 'Insorit'],
        
                            2: ['Normala', 'Mica']}),
        
         'NU': defaultdict(<class 'list'>,
        
                           {0: ['Rece', 'Cald'],
        
                            1: ['Innorat', 'Insorit'],
        
                            2: ['Mare', 'Mare']})
        
        }
            
            Acum că avem dicționarul ce conține apariția tuturor atributelor/feature-urilor în funcție de clasă și tipul
             feature-ului, trebuie să obținem probabilitățile relative ale acestora.
    Pentru a realiza acest lucru, o să iterăm prin intregul dicționar și pentru fiecare cheie de feature, o să înlocuim
    valoarea acestuia (lista) cu dicționarul returnat de self._calculate_relative_proba .
    Forma finală a lui self.nb_dict o să fie:{
        'DA': defaultdict(<class 'list'>,
        
                           {0: {'Cald': 0.5, 'Placut': 0.5},
        
                            1: {'Insorit': 0.5, 'Senin': 0.5},
        
                            2: {'Mica': 0.5, 'Normala': 0.5}}),
        
         'NU': defaultdict(<class 'list'>,
        
                           {0: {'Cald': 0.5, 'Rece': 0.5},
        
                            1: {'Innorat': 0.5, 'Insorit': 0.5},
        
                            2: {'Mare': 1.0}})
    
    }
    Dicționarul de mai sus, este practic, informația invățată de clasificatorul nostru.
        '''
        for label in self.labels:
            X_class = []  # a list which will contain all the examples for a specific class/label
            for example_index, example_label in enumerate(y):
                if example_label == label:
                    X_class.append(X[example_index])

            examples_class_no, features_class_no = np.shape(X_class)

            for feature_index in range(features_class_no):
                for item in X_class:
                    self.nb_dict[label][feature_index].append(item[feature_index])

        # Now we have a dictionary containing all occurences of feature values, per feature, per class
        # We need to transform this dict to a dict with relative feature value probabilities per class
        for label in self.labels:
            for feature_index in range(features_no):
                self.nb_dict[label][feature_index] = self._calculate_relative_proba(self.nb_dict[label][feature_index])



    '''
        Parametrul X_new este exemplul ce urmează să fie clasificat.
    Se crează dicționarul Y_dict pentru a putea păstra probabilitățile de apartenență(scorurile) ale exemplului pentru fiecare clasă.
    Se extrag valorile probabilităților condiționate pentru fiecare tip de feature(x1,x2...xn).
    Se verifică pentru fiecare feature din X_new dacă are calculată probabilitatea condiționată(dacă există în nb_dict).
     În cazul în care feature-ul are deja calculată probabilitatea, se înmulțește valoarea acesteia cu restul probabilităților condiționate ale celorlalte feature-uri și cu probabilitatea pentru clasa (P(y)).
    Dacă feature-ul nu are calculată probabilitatea, rezultatul se va înmulți cu zero.
     La finalul celui de al-2-lea for, se adaugă valoarea calculată în Y_dict.
     În final, se returnează clasa de apartenență a noului exemplu aplând metoda _get_class().
    
    '''
    def predict(self, X_new):
        Y_dict = {}

        # First we determine the class-probability of each class, and then we determine the class with the highest probability
        for label in self.labels:
            class_probability = self.class_probabilities[label]

            for feature_index in range(len(X_new)):
                relative_feature_values = self.nb_dict[label][feature_index]
                if X_new[feature_index] in relative_feature_values.keys():
                    class_probability *= relative_feature_values[X_new[feature_index]]
                else:
                    class_probability *= 0.01  # Lidstone smoothing
            Y_dict[label] = class_probability

        return self._get_class(Y_dict)

    '''metoda pentru initializare a dictionarului(de memorare a informatiilor calculate in procesul de antrenare)
        np_dict = realizam un dictionar, ce contine la fiecare cheie un default dictionary de tip lista
        self.labels-> np.array, elem unice, contine setul de antrenare
        #ex: self.labels = ['Da', 'Nu'], iar nb_dict= { 'Da': defaultdict(<list>), 'Nu': defaultdict(<list>) }
    '''
    def _initialize_nb_dict(self):
        self.nb_dict = {}
        for label in self.labels:
            self.nb_dict[label] = defaultdict(list)

    '''metoda pentru calcularea probabilitatilor de aparitie relative/ prob de aparitie 
        a claselor.
        
        la input = primeste o lista de feature-uri pt exemplele din setul de antrenare ce apartin
        uneia din clasele elements_list['Rece', 'Cald'], feature-urile de Temperatura(x1), pentru clasa 'NU').
        - daca dorim sa calculam probabilitatile de aparitie a claselor, lista trb sa contina label-urile pt 
        exemplele din setul de antrenare.
        
        Counter = este din "collecctions" si ne ajuta sa calculam frecventa de aparitie a elementelor dintr-o lista.
        , el ne va intoarce un dictionar unde <key> = elementul din lista, <val> = nr lui de aparitie in lista
        
        II dai: elements_list = ['Rece', 'Rece", 'Cald']
        si iti scoate 'Rece' : 2/3 % , 'Cald': 1/3 %
        P = Cate de 'Rece' este in lista / numarul de lemente
        '''
    @staticmethod
    def _calculate_relative_proba(elements_list):
        no_examples = len(elements_list)
        occurrence_dict = dict(Counter(elements_list))

        for key in occurrence_dict.keys(): #deci pt fiecare cheie, valoarea o sa fie probabilitatea de aparitie
            occurrence_dict[key] = occurrence_dict[key] / float(no_examples)

        return occurrence_dict
    '''
    Metoda pentru selectia clasei in functie de scorul cel mai mare
        Metoda folosita in metoda predict()
        - ne ajuta sa decidem clasa, pe baza probabilitatilor calculate cu Theorema Bayes(clasa cu scorul
        cel mai mare, devine clasa exemplului pe care dorim sa il clasificam)
        
        input: score_dict (este un dictionar ce contine clasele pe post de chei si probabilitatile calculate cu
        Bayes pe post de valori.
    
    
    '''
    @staticmethod
    def _get_class(score_dict):
        sorted_dict = sorted(score_dict.items(), key=lambda value: value[1], reverse=True) #sortam descrescator dupa valuare
        sorted_dict = dict(sorted_dict)
        keys_list = list(sorted_dict.keys())
        max_key = keys_list[0]
        return max_key