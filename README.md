# RecSys


## Uvod 
Ovaj repozitorijum sadrži algoritme iz bilbioteke **Surprise** (https://surprise.readthedocs.io/en/stable/index.html), koji rade sa podacima tj. skupom podataka **R4**, koji je obezbeđen od strane **Research Yahoo! Aliance** (https://tinyurl.com/y2aqxo99).

## Podaci 
Skup podataka pod nazivom R4 je dostupan u sklopu **Yahoo! Research Alliance Webscope  programa** i sme se koristiti samo u svrhe nekomercijalnih istraživanja.  On sadrži mali uzorak ocena različitih filmova od strane korisnika **Yahoo! Movies zajednice**. Skup takođe obuhvata informacije o velikom broju filmova koji su se pojavili do novembra 2003. godinе (sinopsis, lista glumaca, lista žanrova kojima film pripada, lista producenata itd).
Podaci su raspoređeni u šest fajlova (folder R4):
- movie_db_yoda
- readme
- WebscopeReadMe
- ydata-ymovies-mapping-to-eachmovie-v1_0.txt
- ydata-ymovies-mapping-to-movielens-v1_0.txt
- ydata-ymovies-user-demographics-v1_0.txt
- ydata-ymovies-user-movie-ratings-train-v1_0.txt
- ydata-ymovies-user-movie-ratings-test-v1_0.txt
Celokupan opis svih fajlova možete pronaći u **readme** datoteci.

Datoteka *movie_db_yoda* sadrži podatke o filmovima (id filma, naslov, sinopsis, listu glumaca, listu žanrova itd.)
Trening skup je smešten u fajlu *ydata-ymovies-user-movie-ratings-train-v1_0.txt*, dok se testni skup nalazi u *ydata-ymovies-user-movie-ratings-test-v1_0.txt*. Oba fajla sadrže id filma, id korisnika, rejting (vrednost na skali od 1 do 13) i *konvertovani* rejting  skaliran na vrednosti od 1 do 5. Prilikom implementacije svih algoritama korišćena je prva skala. 

Datoteka *ydata-ymovies-user-demographics-v1_0.txt* sadrži podatke o polu i godini rođenja korisnika (id korisnika, godina rođenja i pol).

U folder R4 dodat je novi fajl *fullSet.txt*, koji obuhvata podatke o rejtinzima iz trening i testnog skupa. 

## Rad sa podacima
Potrebne funkcije za rada sa podacima su implementirane u klasi YahooDataset. Konkretnije, trening i testni skup se učitavaju pomoću metode **loadYahooDataset()**, dok se podaci iz fajla *fullSet.txt* učitavaju pozivom metode **loadFullSet()**.

## Pokretanje
U nastavku se navodi lista fajlova koje možete pokrenuti:
- exploration.py
- k-fold validation.py
- BayesianPersonalizedRanking
- CollaborativeRecommenders-Accuracy
- ContentKNNRecommenders-Accuracy
- SVD-Accuracy
- WeightedHybrid-Accuracy


Moguće je pokrenut sve skripte koje u svom nazivu ne sadrže "Algorithm", osim YahooDataset.py (definicija klase za učitavanje dataseta). 
U folderu R4 su smešteni dataset fajlovi. U direktorijumu Slike nalaze se slike dobijene pokretanjem odgovarajućih skripti, kao i skrinšotovi nekih rezultata evaluacije (top-n 



Kako biste se upoznali sa datasetom, najbolje je prvo pokrenuti exploration.py, a zatim k-fold validation. Skripta k-fold validation učitava ceo skup (podatke iz trening i testnog skupa) i vrši k-fold validaciju algoritama implementiranih u surprise. Pomenuti algoritmi su inicijalizovani sa difoltnim parametrima čije vrednosti možete pronaći u dokumentaciji bibloteke surprise (link koji je naveden iznad).

Skriptom BayesianPersonalizedRanking ispisuje se prosečna vrednost AUC metrike, pošto je ovaj algoritam u suštini klasifikator (da li će se korisniku dopasti ili neće određeni proizvod). Takođe, pokretanjem ove skripte generišu se i standardne metrike (hit rate, cumulative hit rate, average reciprocal hit rate itd). Naravno, što su ove vrednosti veće od nule, to je lista koju algoritam generiše "kvalitetnija". Parametre algoritma možete menjati izmenom vrednosti u promenljivi bpr_params. 

Skripta CollaborativeRecommenders-Accuracy samo kreira grafikon koji pokazuje zavisnost Root Mean Square Error metrike od broja suseda u knn algoritmima koje možete pronaći na linku (https://surprise.readthedocs.io/en/stable/knn_inspired.html). 

Skripta ContentKNNRecommenders-Accuracy ispituje RMSE metrike ContentKNN algoritma. Funkcijom loadYahooDataset() se učitaju trening i testni skup, zatim sledi treniranje pomenutog algoritma. Zatim sledi testiranje algoritma na testnom skupu.
Promenljiva predictions sadrži listu predikcija tj. listu uređenih četvorki (id korisnika, id filma, ostavljeni rejting, predikcija rejinga).
Na kraju, jednostavno se ispiše RMSE metrika.

Opisani princip u delu ContentKNNRecommenders-Accuracy je isti i za ostale skripte koji u svom nazivu imaju -Accuracy.

Fajl top-n evaluation.py ispisuje Recommender metrike za top-n listu za većinu algoritama. Od linije 38 pa do 42 su inicijalizovani algoritmi. Na liniji 44 referenci algo se dodeljuje  refernca algoritma za koga želimo da dobijemo metrike.

knnRecAlgorithm je zapravo implementacija user collaborative filtering algoritma. 
Skritpa knn-top n evaluaton.py  ispisuje Recommender metrike za top-n listu za user collaborative algoritam tj. knnRecAlgorithm
