# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 10:47:39 2020

@author: Jovic
"""

from YahooDataset import YahooDataset
from RecommenderMetrics import RecommenderMetrics
from surprise import SVD

import random
import numpy as np

np.random.seed(0)
random.seed(0)

yd = YahooDataset()
testGlobalMean = yd.getTestDataGlobalMean()

svd = SVD(random_state = 0, reg_all=0.1, lr_all=0.003, n_factors=30, verbose=False)
for trainSet, testSet in yd.loadYahooDataset():
    trainSet.rating_scale = (1, 13)
    svd.fit(trainSet)
    predictions = svd.test(testSet)
    RecommenderMetrics.RMSE(predictions)
    RecommenderMetrics.RRSE(predictions, testGlobalMean)