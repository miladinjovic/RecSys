# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 15:30:55 2020

@author: Jovic
"""
from YahooDataset import YahooDataset
from surprise.model_selection import LeaveOneOut
from RecommenderMetrics import RecommenderMetrics
from knnRecAlgorithm import knnRecAlgorithm
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


LOOCV = LeaveOneOut(n_splits=1)
algo = knnRecAlgorithm()

for trainSet, testSet in LOOCV.split(data):
    trainSet.rating_scale = (1, 13)
    algo.fit(trainSet)
    topNPredicted = defaultdict(list)
    
    leftOutPredictions = algo.test(testSet)
    
    for user in range(trainSet.n_users):
        userID = int(trainSet.to_raw_uid(user))
        antiTestSet = buildAntiTestSetForUser(user, trainSet)
        predictions = algo.test(antiTestSet)
        
        topNPredicted[userID]=RecommenderMetrics.GetTopNForUser(predictions, minimumRating=0.0)
#    
#    for user in range(trainSet.n_users):
#        userID = int(trainSet.to_raw_uid(user))
#        topNPredicted[userID] = algo.recommend(user)
#        
    #Evaluation
        
    print("\nHit Rate: ", RecommenderMetrics.HitRate(topNPredicted, leftOutPredictions))
    print("\ncHR (Cumulative Hit Rate, rating >= 8): ", RecommenderMetrics.CumulativeHitRate(topNPredicted, leftOutPredictions))
    print("Rating hit rate\n")
    RecommenderMetrics.RatingHitRate(topNPredicted, leftOutPredictions)
    print("\nARHR (Average Reciprocal Hit Rank): ", RecommenderMetrics.AverageReciprocalHitRank(topNPredicted, leftOutPredictions))
#    
        
#print("\nComputing complete recommendations, no hold outs...")
#
#fullTrainSet = data.build_full_trainset()
#fullTrainSet.rating_scale = (1, 13)
#algo.fit(fullTrainSet)
##sim_options = {'name': 'pearson_baseline', 'user_based': False}
##simsAlgo = KNNBaseline(sim_options=sim_options)
##simsAlgo.fit(fullTrainSet)
#
#topNPredicted = defaultdict(list)
#for user in range(fullTrainSet.n_users):
#    userID = int(fullTrainSet.to_raw_uid(user))
#    antiTestSet = buildAntiTestSetForUser(user, fullTrainSet)
#    predictions = algo.test(antiTestSet)
#    
#    topNPredicted[userID]=RecommenderMetrics.GetTopNForUser(predictions)
#
#
#print("\nUser coverage: ", RecommenderMetrics.UserCoverage(topNPredicted, fullTrainSet.n_users))
#
##print("\nDiversity: ", RecommenderMetrics.Diversity(topNPredicted, simsAlgo))
#
#rankings, _, _, _ = yd.loadMovies()
#
#print("\nNovelty (average popularity rank): ", RecommenderMetrics.Novelty(topNPredicted, rankings))

