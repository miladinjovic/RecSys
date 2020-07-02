# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 17:18:07 2019

@author: Jovic
"""

from YahooDataset import YahooDataset
from RecommenderMetrics import RecommenderMetrics
from ContentKNNAlgorithm import ContentKNNAlgorithm

import random
import numpy as np

np.random.seed(0)
random.seed(0)

yd = YahooDataset()

algo = ContentKNNAlgorithm()
for trainSet, testSet in yd.loadYahooDataset():
    algo.fit(trainSet)
    predictions = algo.test(testSet)
    RecommenderMetrics.RMSE(predictions)
