# RecSys


## Uvod 
Ovaj repozitorijum sadrži implementaciju sistema za preporuku pomoću bilbioteke **Surprise** (https://surprise.readthedocs.io/en/stable/index.html). 
## Podaci 
Skup podataka pod nazivom R4 je dostupan u sklopu **Yahoo! Research Alliance Webscope  programa** i sme se koristiti samo u svrhe nekomercijalnih istraživanja.  On sadrži mali uzorak ocena različitih filmova od strane korisnika **Yahoo! Movies zajednice**. Skup takođe obuhvata informacije o velikom broju filmova koji su se pojavili do novembra 2003. godinе (sinopsis, lista glumaca, lista žanrova kojima film pripada, lista producenata itd).
Podaci su raspoređeni u šest fajlova (**folder R4**):
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

U folder R4 dodat je novi fajl *fullSet.txt*, koji obuhvata podatke o rejtinzima iz trening i testnog skupa, radi lakše implemenacije i evaluacije pojedinih algoritama. 

## Rad sa podacima
Potrebne funkcije za rad sa podacima su implementirane u klasi YahooDataset. Konkretnije, trening i testni skup se učitavaju pomoću metode **loadYahooDataset()**, dok se podaci iz fajla *fullSet.txt* učitavaju pozivom metode **loadFullSet()**.

## Aplikacija
U folderu **Aplikacija** se nalaze sve skripte, od kojih je moguće pokrenuti:
- exploration.py
- k-fold validation.py
- precisionAndRecall.py
- BayesianPersonalizedRanking.py
- CollaborativeRecommenders-Accuracy.py
- ContentKNNRecommenders-Accuracy.py
- SVD-Accuracy.py
- WeightedHybrid-Accuracy.py
- top-n evaluation.py
- knn - top n evaluation.py


Kako biste se upoznali sa datasetom, najbolje je prvo pokrenuti **exploration.py**, a zatim **k-fold validation.py**. Skripta k-fold validation.py učitava ceo skup (podatke iz trening i testnog skupa) i vrši k-fold validaciju algoritama iz biblioteke surprise. Pomenuti algoritmi su inicijalizovani difoltnim parametrima čije vrednosti možete pronaći u dokumentaciji bibloteke surprise (https://surprise.readthedocs.io/en/stable/prediction_algorithms_package.html).

Skriptom **BayesianPersonalizedRanking.py** se ispisuje prosečna vrednost AUC metrike, pošto je ovaj algoritam u suštini klasifikator (da li će se korisniku dopasti ili neće određeni proizvod). Takođe, pokretanjem ove skripte generišu se i standardne metrike top-n evaluacije (hit rate, cumulative hit rate, average reciprocal hit rate itd). Naravno, što su ove vrednosti veće od nule, to je lista koju algoritam generiše "kvalitetnija". Parametre algoritma možete menjati izmenom vrednosti u promenljivi bpr_params. 

Skripta **CollaborativeRecommenders-Accuracy.py** samo kreira grafikon koji pokazuje zavisnost Root Mean Square Error metrike od broja suseda u knn algoritmima koje možete pronaći na linku (https://surprise.readthedocs.io/en/stable/knn_inspired.html). 

Skripta **ContentKNNRecommenders-Accuracy.py** računa RMSE (**R**oot **M**ean **S**quare **E**rror) metriku ContentKNN algoritma. Prvo se učitaju trening i testni skup, zatim se obavlja obučavanje algoritma na osnovu trening skupa. Nakon toga, sprovodi se evaluacija pomoću testnog skupa.
Promenljiva **predictions** sadrži listu predikcija tj. listu uređenih četvorki (id korisnika, id filma, ostavljeni rejting, predikcija rejinga).
Opisani princip rada je isti i u skriptama **SVD-Accuracy.py** i **WeightedHybrid-Accuracy.py**.

Fajl **top-n evaluation.py** ispisuje Recommender metrike za top-n listu za većinu algoritama. Na linijama od 38. do 42. inicijalizovani su različiti algoritmi. Na liniji 44 referenci algo je potrebno dodeliti referencu na algoritam za koga želimo da dobijemo top-n metrike.

Datoteka **knnRecAlgorithm** je implementacija user collaborative filtering algoritma. 
Skritpa **knn-top n evaluaton.py**  ispisuje Recommender metrike top-n liste za user collaborative algoritam, koji je definisan klasom **knnRecAlgorithm**

## Napomena
U folderu **Slike** smešteni su su dijagrami dobijeni pokretanjem skripte exploration.py, kao i skinšotovi rezultata top-n evaluacije različitih algoritama. Datoteka **requirements.txt** sadrži spisak svih potrebnih biblioteka, koji će vam pomoći da lako kreirate venv na vašem računaru.
U toku je izrada veb aplikacije  koja se zasniva na ovoj aplikaciji. Link do repozitorijuma pomenute veb aplikacije je https://github.com/miladinjovic/RecSys-GUI
