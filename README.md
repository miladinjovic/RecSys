# RecSys


Moguće je pokrenut sve skripte koje u svom nazivu ne sadrže "Algorithm", osim YahooDataset.py (definicija klase za učitavanje dataseta). 
U folderu R4 su smešteni dataset fajlovi. U direktorijumu Slike nalaze se slike dobijene pokretanjem odgovarajućih skripti, kao i skrinšotovi nekih rezultata evaluacije (top-n metoda).
Korišćeni algoritmi iz biblioteke Surprise (https://surprise.readthedocs.io/en/stable/index.html).

Trening i testni skup se učitavaju pomoću metode loadYahooDataset() klase YahooDataset. Oni su već unapred kreirani i dobijaju se prilikom preuzimanja skupa R4.

Kako biste se upoznali sa datasetom, najbolje je prvo pokrenuti exploration.py, a zatim k-fold validation. Skripta k-fold validation učitava ceo skup (podatke iz trening i testnog skupa) i vrši k-fold validaciju algoritama implementiranih u surprise. Pomenuti algoritmi su inicijalizovani sa difoltnim parametrima čije vrednosti možete pronaći u dokumentaciji bibloteke surprise (link koji je naveden iznad).

Skriptom BayesianPersonalizedRanking ispisuje se prosečna vrednost AUC metrike, pošto je ovaj algoritam u suštini klasifikator (da li će se korisniku dopasti ili neće određeni proizvod). Takođe, pokretanjem ove skripte generišu se i standardne metrike (hit rate, cumulative hit rate, average reciprocal hit rate itd). Naravno, što su ove vrednosti veće od nule, to je lista koju algoritam generiše "kvalitetnija". Parametre algoritma možete menjati izmenom vrednosti u promenljivi bpr_params. 

Skripta CollaborativeRecommenders-Accuracy samo kreira grafikon koji pokazuje zavisnost Root Mean Square Error metrike od broja suseda u knn algoritmima koje možete pronaći na linku (https://surprise.readthedocs.io/en/stable/knn_inspired.html). 

Skripta ContentKNNRecommenders-Accuracy ispituje RMSE metrike ContentKNN algoritma. Funkcijom loadYahooDataset() se učitaju trening i testni skup, zatim sledi treniranje pomenutog algoritma. Zatim sledi testiranje algoritma na testnom skupu.
Promenljiva predictions sadrži listu predikcija tj. listu uređenih četvorki (id korisnika, id filma, ostavljeni rejting, predikcija rejinga).
Na kraju, jednostavno se ispiše RMSE metrika.

Opisani princip u delu ContentKNNRecommenders-Accuracy je isti i za ostale skripte koji u svom nazivu imaju -Accuracy.

Fajl top-n evaluation.py ispisuje Recommender metrike za top-n listu za većinu algoritama. Od linije 38 pa do 42 su inicijalizovani algoritmi. Na liniji 44 referenci algo se dodeljuje  refernca algoritma za koga želimo da dobijemo metrike.

knnRecAlgorithm je zapravo implementacija user collaborative filtering algoritma. 
knn-top n evaluaton.py  ispisuje Recommender metrike za top-n listu za user collaborative algoritam tj. knnRecAlgorithm
