# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 12:25:30 2020

@author: Jovic
"""

from YahooDataset import YahooDataset
from surprise import SVD
from surprise import KNNBaseline, KNNWithMeans
from RecommenderMetrics import RecommenderMetrics
from WeightedHybridAlgorithm import WeightedHybridAlgorithm

import random
import numpy as np

np.random.seed(0)
random.seed(0)

yd = YahooDataset()
svd = SVD(random_state = 0, reg_all=0.1, lr_all=0.003, n_factors=30, verbose=False)
knn = KNNBaseline(sim_options = {'name': 'cosine', 'user_based': False}, k = 150)
weightedHybrid = WeightedHybridAlgorithm(svd, knn, weights=[0.6, 0.4])


for trainSet, testSet in yd.loadYahooDataset():
    trainSet.rating_scale = (1, 13)
    weightedHybrid.fit(trainSet)
    predictions = weightedHybrid.test(testSet)
#    RecommenderMetrics.RRSE(predictions, yd.getTestDataGlobalMean())
    RecommenderMetrics.RMSE(predictions)