# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 16:30:22 2020

@author: Jovic
"""
import warnings
warnings.simplefilter("ignore")

from YahooDataset import YahooDataset
from surprise import KNNBaseline
from bprRecommenderMetrics import bprRecommenderMetrics
from BPRAlgorithm import BPRAlgorithm
from scipy.sparse import csr_matrix, dok_matrix
import numpy as np
from math import ceil
from sklearn.metrics import roc_auc_score

"""http://ethen8181.github.io/machine-learning/recsys/4_bpr.html"""

np.random.seed(0)

def create_matrix(data):
    
    data['bprRating'] = np.where(data['rating']>=0, 1, 0)
    
    for col in ('movieId', 'userId', 'bprRating'):
        data[col]= data[col].astype('category')
    
    data["codes"] = data['movieId'].cat.codes
    ratings = csr_matrix((data['bprRating'],(data['userId'].cat.codes, data['movieId'].cat.codes)))
    ratings.eliminate_zeros()
    return ratings, data

def create_train_test(ratings, test_size = 0.2, seed = 1234):
    assert test_size < 1.0 and test_size > 0.0
    train = ratings.copy().todok()
    test = dok_matrix(train.shape)
    
    rstate = np.random.RandomState(seed)
    for u in range(ratings.shape[0]):
        split_index = ratings[u].indices
        if(len(split_index) == 1):
            continue
        n_splits = ceil(test_size * split_index.shape[0])
        test_index = rstate.choice(split_index, size = n_splits, replace = False)
        test[u, test_index] = ratings[u, test_index]
        train[u, test_index] = 0
    
    train, test = train.tocsr(), test.tocsr()
    return train, test

    
def leaveOneOut_split(ratings, seed = 1234, df = None):
    train = ratings.copy().todok()
    test = dok_matrix(train.shape)
    
    rstate = np.random.RandomState(seed)
    for u in range(ratings.shape[0]):
        if(len(ratings[u].indices) == 1 ):
            continue
        userId = df["userId"].cat.categories[u]
        split_index = df[df["userId"] == userId]["codes"].values
        test_index = rstate.choice(split_index, size = 1, replace = False)
        test[u, test_index] = ratings[u, test_index]
        train[u, test_index] = 0
    
    train, test = train.tocsr(), test.tocsr()
    yield train, test


def auc_score(model, ratings):
    auc = 0.0
    n_users, n_items = ratings.shape
    for user, row in enumerate(ratings):
        if(len(row.indices) == 0):
            n_users -= 1
            continue
        y_pred = model._predict_user(user)
        y_true = np.zeros(n_items)
        y_true[row.indices] = 1
        auc += roc_auc_score(y_true, y_pred)

    auc /= n_users
    return auc


yd = YahooDataset()
df = yd.loadYahooPandasFullDataFrame()
ratingCol = df["rating"]
df = yd.normalizeByUser(df)

X, df = create_matrix(df)
df["rating"] = ratingCol


X_train, X_test = create_train_test(X, test_size = 0.2, seed = 1234)


bpr_params = {'reg': 0.02,
              'learning_rate': 0.2,
              'n_iters': 600,  
              'n_factors': 15,
              'batch_size': 100,
              'df': df}


bpr = BPRAlgorithm(**bpr_params)
bpr.fit(X_train)
print("Mean AUC score per user: %f"  % auc_score(bpr, X_test))

for trainSet, testSet in leaveOneOut_split(X, seed = 1234, df = df): 
    bpr.fit(trainSet)
    recs = bpr.recommend(trainSet, excluded=testSet)
    print("\nHit Rate: ", bprRecommenderMetrics.HitRate(recs, testSet))
    print("\ncHR (Cumulative Hit Rate, rating >= 8): ", bprRecommenderMetrics.CumulativeHitRate(recs, testSet, df = df))
    print("Rating hit rate\n")
    bprRecommenderMetrics.RatingHitRate(recs, testSet, df)
    print("Average Reciprocal Hit Rank: ", bprRecommenderMetrics.AverageReciprocalHitRank(recs, testSet))
    

#print("\nComputing complete recommendations, no hold outs...")

#fullTrainSet = yd.loadFullSet().build_full_trainset()
#fullTrainSet.rating_scale = (1, 13)
#sim_options = {'name': 'pearson_baseline', 'user_based': False}
#simsAlgo = KNNBaseline(sim_options = sim_options)
#simsAlgo.fit(fullTrainSet)
#
#bpr.fit(X)
#topNPredicted = bpr.recommend(X)
#
#print("\nDiversity: ", bprRecommenderMetrics.Diversity(topNPredicted, simsAlgo, df = df))
#
#rankings, _, _, _ = yd.loadMovies()
#
#print("\nNovelty (average popularity rank): ", bprRecommenderMetrics.Novelty(topNPredicted, rankings, df = df))