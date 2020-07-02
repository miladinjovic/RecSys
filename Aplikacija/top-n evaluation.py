# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 13:34:30 2020

@author: Jovic
"""

from YahooDataset import YahooDataset
from surprise.model_selection import LeaveOneOut
from RecommenderMetrics import RecommenderMetrics
from ContentKNNAlgorithm import ContentKNNAlgorithm
from surprise import SVD
from surprise import KNNBaseline
from WeightedHybridAlgorithm import WeightedHybridAlgorithm
from collections import defaultdict

import random
import numpy as np

np.random.seed(0)
random.seed(0)

def buildAntiTestSetForUser(testSubject, trainset):
    fill = trainset.global_mean

    anti_testset = []
        
    user_items = set([j for (j, _) in trainset.ur[testSubject]])
    anti_testset += [(trainset.to_raw_uid(testSubject), trainset.to_raw_iid(i), fill) for
                             i in trainset.all_items() if
                             i not in user_items]
    return anti_testset
    

yd = YahooDataset()
data = yd.loadFullSet()

svd = SVD(random_state = 0, reg_all=0.1, lr_all=0.003, n_factors=30, verbose=False)
#svd = SVD(n_factors= 35, lr_all= 0.002, reg_all= 0.009000000000000001)
knn = KNNBaseline(sim_options = {'name': 'cosine', 'user_based': False}, k=150)
wh = WeightedHybridAlgorithm(svd, knn, [0.7,0.3])
contentKNN = ContentKNNAlgorithm()

algo = contentKNN

LOOCV = LeaveOneOut(n_splits=1)


for trainSet, testSet in LOOCV.split(data):
    trainSet.rating_scale = (1, 13)
    algo.fit(trainSet)
    topNPredicted = defaultdict(list)
    
    leftOutPredictions = algo.test(testSet)
    
    for user in range(trainSet.n_users):
        userID = int(trainSet.to_raw_uid(user))
        antiTestSet = buildAntiTestSetForUser(user, trainSet)
        predictions = algo.test(antiTestSet)
        
        topNPredicted[userID]=RecommenderMetrics.GetTopNForUser(predictions)
    
    #Evaluation
        
    print("\nHit Rate: ", RecommenderMetrics.HitRate(topNPredicted, leftOutPredictions))
    print("\ncHR (Cumulative Hit Rate, rating >= 8): ", RecommenderMetrics.CumulativeHitRate(topNPredicted, leftOutPredictions))
    print("Rating hit rate\n")
    RecommenderMetrics.RatingHitRate(topNPredicted, leftOutPredictions)
    print("\nARHR (Average Reciprocal Hit Rank): ", RecommenderMetrics.AverageReciprocalHitRank(topNPredicted, leftOutPredictions))
#    
        
print("\nComputing complete recommendations, no hold outs...")

fullTrainSet = data.build_full_trainset()
fullTrainSet.rating_scale = (1, 13)
#algo.fit(fullTrainSet)
#sim_options = {'name': 'pearson_baseline', 'user_based': False}
#simsAlgo = KNNBaseline(sim_options=sim_options)
#simsAlgo.fit(fullTrainSet)

topNPredicted = defaultdict(list)
for user in range(fullTrainSet.n_users):
    userID = int(fullTrainSet.to_raw_uid(user))
    antiTestSet = buildAntiTestSetForUser(user, fullTrainSet)
    predictions = algo.test(antiTestSet)
    
    topNPredicted[userID]=RecommenderMetrics.GetTopNForUser(predictions)


print("\nUser coverage: ", RecommenderMetrics.UserCoverage(topNPredicted, fullTrainSet.n_users))
#
#print("\nDiversity: ", RecommenderMetrics.Diversity(topNPredicted, simsAlgo))

rankings, _, _, _ = yd.loadMovies()

print("\nNovelty (average popularity rank): ", RecommenderMetrics.Novelty(topNPredicted, rankings))