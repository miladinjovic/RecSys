import pandas as pd
import os
import csv
import sys

from surprise import Dataset
from surprise import Reader
from surprise.model_selection import PredefinedKFold

from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
# import nltk
from nltk.corpus import stopwords

class YahooDataset:

    # movieID_to_name = {}
    # name_to_movieID = {}
    
    ratingsTrainPath = '../R4/ydata-ymovies-user-movie-ratings-train-v1_0.txt'
    ratingsTestPath = '../R4/ydata-ymovies-user-movie-ratings-test-v1_0.txt'
    usersPath = '../R4/ydata-ymovies-user-demographics-v1_0.txt'
    moviesPath = '../R4/movie_db_yoda'
    fullSetPath = '../R4/fullSet.txt'
    
    def loadFullSet(self):
        reader = Reader(line_format='user item rating timestamp' ,sep='\t', skip_lines=0)
        fullDataset = Dataset.load_from_file(self.fullSetPath, reader=reader)

        return fullDataset
    
    def loadYahooPandasFullDataFrame(self):
        # os.chdir(os.path.dirname(sys.argv[0]))
        data = pd.read_csv(self.fullSetPath, delimiter = '\t', header=None, 
                          names = ["userId", "movieId","rating"], usecols =["userId", "movieId", "rating"])
        return data
    
    
#    def loadMoviePandas(self):
#        data = pd.read_csv(self.moviesPath, encoding = "ISO-8859-1", delimiter='\t', header=None)
#        return data
    
    # def loadDemographicsData(self):
    #     demographics = {}
    #     with open(self.usersPath, newline='') as csvfile:
    #         demographicReader = csv.reader(csvfile, delimiter='\t')
    #         for row in demographicReader:
    #             userId = int(row[0])
                
    #             if(row[1] == "undef"):
    #                 year = 0
    #             else:
    #                 year = int(row[1])
                
    #             gender = row[2]
    #             demographics[userId] = {"year": year, "gender": gender }
                
    #     return pd.DataFrame(demographics).T.reset_index().rename(columns={"index": "userId"})
    
    
    def loadDemographicsData(self):
        # os.chdir(os.path.dirname(sys.argv[0]))
        data = pd.read_csv(self.usersPath, delimiter = '\t', header=None, 
                          names = ["userId", "year","gender"])
        return data
    
    def loadNormalizedData(self, data=None):
        if data is None:
            data = self.loadYahooPandasTrainingDataframe()
        
        normalizedByUser, usersAverage = self.normalizeByUser(data)
        
        allNormalized, itemsAverage = self.normalizeByItem(normalizedByUser)
        
        return (allNormalized, usersAverage, itemsAverage)
        
    
    def normalizeByUser(self, data=None ):
        if data is None:
            data = self.loadYahooPandasTrainingDataframe()
        
        #data.rating = data.iloc[:, [0, 2]].set_index("userId").transform(lambda p: p-usersAverage.loc[p.name, "rating"] ,axis=1).reset_index().rating
#        usersAverage =  data.iloc[:, [0, 2]].groupby("userId").mean()
        normalizedByUser = data.set_index("movieId").groupby("userId").transform(lambda p: p-p.mean()).reset_index()
        normalizedByUser.insert(0, "userId", data.userId)
        
        return normalizedByUser

    
    def normalizeByItem(self, data=None):
        if data is None:
            data = self.loadYahooPandasTrainingDataframe()
        
#        itemsAverage = data.iloc[:, [1,2]].groupby("movieId").mean()
        normalizedByItem = data.set_index("userId").groupby("movieId").transform(lambda p: p-p.mean()).reset_index()
        normalizedByItem.insert(1, "movieId", data.movieId)
        
        return normalizedByItem
    
    def loadFromPandas(self, frame):
         reader = Reader(rating_scale=(frame["rating"].min(), frame["rating"].max()))
         data = Dataset.load_from_df(frame, reader=reader)
         return data
        
        
    def loadYahooPandasTrainingDataFrame(self):
        # os.chdir(os.path.dirname(sys.argv[0]))
        data = pd.read_csv(self.ratingsTrainPath, delimiter = '\t', header=None, 
                          names = ["userId", "movieId","rating"], usecols =["userId", "movieId", "rating"])
        return data
    
    def loadYahooPandasTestDataFrame(self):
        # os.chdir(os.path.dirname(sys.argv[0]))
        data = pd.read_csv(self.ratingsTestPath, delimiter = '\t', header=None, 
                          names = ["userId", "movieId","rating"], usecols =["userId", "movieId", "rating"])
        return data
    
    def getTestDataGlobalMean(self):
        testData = self.loadYahooPandasTestDataFrame()
        return testData["rating"].mean()
    
    def loadYahooDataset(self):
        
        # os.chdir(os.path.dirname(sys.argv[0]))
        reader = Reader(line_format='user item rating timestamp' ,sep='\t', skip_lines=0)
        
        data = Dataset.load_from_folds([(self.ratingsTrainPath, self.ratingsTestPath)], reader=reader)
        pkf = PredefinedKFold()
        
        return pkf.split(data)
    
    def loadYahooTrainDataset(self):

        # os.chdir(os.path.dirname(sys.argv[0]))
        reader = Reader(line_format='user item rating timestamp' ,sep='\t', skip_lines=0)
        ratingsTrainDataset = Dataset.load_from_file(self.ratingsTrainPath, reader=reader)

        return ratingsTrainDataset.build_full_trainset()
    
    def loadYahooTestDataset(self):
        
         # os.chdir(os.path.dirname(sys.argv[0]))
         reader = Reader(line_format='user item rating timestamp' ,sep='\t', skip_lines=0)
         ratingsTestDataset = Dataset.load_from_file(self.ratingsTestPath, reader=reader)
         
         return ratingsTestDataset.construct_testset(ratingsTestDataset.raw_ratings)
    
    def loadMovies(self):
        rankings = defaultdict(int)
        gnpp = defaultdict(int)
        genres = defaultdict(list)
        actors = defaultdict(list)
        actorIDs = {}
        genreIDs = {}
        docs = []
        movieIDS = []
        maxGenreID = 0
        maxActorID = 0
                         
        # nltk.download('stopwords')
        stopwords_list = stopwords.words('english')
        
        vectorizer = TfidfVectorizer(analyzer='word',
                          token_pattern = "[a-z]+",
                          ngram_range=(1, 1),
                          # min_df=0.003,
                          # max_df=0.5,
                          max_features=10,
                          stop_words=stopwords_list)
        
        
#        year = defaultdict(int)
        
        with open(self.moviesPath, newline='') as csvfile:
              movieReader = csv.reader(csvfile, delimiter='\t')
              for row in movieReader:
                  movieId = int(row[0])
                  movieIDS.append(movieId)
                 
                  globalPopularity = row[31]
                  genreList = row[10]
                  actorList = row[17]
                  title = row[1]
                  synopsis = row[2]
                 
                  if synopsis == "\\N":
                      synopsis = ""
                     
                  doc = title + " " + synopsis
                  docs.append(doc)
#                 release = row[7]10
#                 if release != "\\N":
#                     year[movieId] = int(release[:4])
                 
                  if globalPopularity !="\\N":
                      globalPopularity = float(globalPopularity)
                      gnpp[movieId] = globalPopularity
                 
                  genreIDList = []
                  if genreList != "\\N":
                    genreList = genreList.split("|")
                    for genre in genreList:
                        if genre in genreIDs:
                            genreID = genreIDs[genre]
                        else:
                            genreID = maxGenreID
                            genreIDs[genre] = genreID
                            maxGenreID += 1
                        genreIDList.append(genreID)
                  genres[movieId] = genreIDList
                 
                  actorIDList = []
                  if actorList != "\\N":
                      actorList = actorList.split("|")
                      for actor in actorList:
                          if actor in actorIDs:
                              actorId = actorIDs[actor]
                          else:
                              actorId = maxActorID
                              actorIDs[actor] = actorId
                              maxActorID += 1
                          actorIDList.append(actorId)
                  actors[movieId] = actorIDList
         
#        Generisi rankove za filmove   
        rank = 1
        for movieId, _ in sorted(gnpp.items(), key=lambda x: x[1], reverse=True):
            rankings[movieId] = rank
            rank+=1
    
        
        genres[-1] = maxGenreID
        actors[-1] = maxActorID
        
        tfidf_matrix = vectorizer.fit_transform(docs)
        tfidf_matrix = pd.DataFrame(tfidf_matrix.todense(), index=movieIDS)
        
        
        return (rankings, genres, actors, tfidf_matrix)
    
    def loadFilteredPandasFullDataFrame(self):
        ratings = self.loadYahooPandasFullDataFrame()
        
        ratingsByUser = ratings.groupby('userId', as_index=False).agg({"rating": "count"})
#        ratingsByMovie = ratings.groupby('movieId', as_index=False).agg({"rating": "count"})

#        ratingsByUser['outlier'] = (abs(ratingsByUser.rating - ratingsByUser.rating.mean()) > ratingsByUser.rating.std() * 3.0)
        ratingsByUser["outlier"] = (ratingsByUser.rating < 5)
        ratingsByUser = ratingsByUser.drop(columns=['rating'])
        combined = ratings.merge(ratingsByUser, on='userId', how='left')
        
#        ratingsByMovie["outlier"] = (ratingsByMovie.rating < 5)
#        ratingsByMovie = ratingsByMovie.drop(columns=['rating'])
#        combined = ratings.merge(ratingsByMovie, on='movieId', how='left')
        
        
        filtered = combined.loc[combined['outlier'] == False]
        filtered = filtered.drop(columns=['outlier'])
        
        return filtered
    
    def loadMovieNames(self):
        movieID_to_name = defaultdict(str)
        with open(self.moviesPath, newline='') as csvfile:
             movieReader = csv.reader(csvfile, delimiter='\t')
             for row in movieReader:
                 movieID = row[0]
                 title = row[1]
                 movieID_to_name[movieID] = title
                 
        return movieID_to_name
    
    def getHistory(self, user, k=10):
        ratings = self.loadYahooPandasFullDataFrame()
        ratingsByUser = ratings[ratings["userId"] == user].sort_values(by=['rating'], ascending = False)
        
        # return ratingsByUser["movieId"][:10]
        return zip(ratingsByUser["movieId"][:10].astype("str"), ratingsByUser["rating"][:10])
    
    
    
    # def loadMovies(self):
    #     # movieID_to_info = defaultdict(str)
    #     movieID_to_info = {}
    #     with open(self.moviesPath, newline='') as csvfile:
    #          movieReader = csv.reader(csvfile, delimiter='\t')
    #          for row in movieReader:
    #              movieID = row[0]
    #              title = row[1]
    #              genreList = row[10]
    #              actorList = row[16]
    #              synopsis = row[2]

    #              movieID_to_info[movieID] = {"Title": title,
    #                                          "Genres": genreList,
    #                                          "Actors": actorList,
    #                                          "Synopsis": synopsis}
                 
    #     return movieID_to_info
    
                 