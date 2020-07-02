# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 17:22:16 2019

@author: Jovic
"""
from YahooDataset import YahooDataset
from surprise import KNNWithMeans, KNNBaseline
from RecommenderMetrics import RecommenderMetrics
import matplotlib.pyplot as plt


import random
import numpy as np

np.random.seed(0)
random.seed(0)


yd = YahooDataset()
kS =[]
accuraciesBaselineItem = []
accuraciesBaselineUser = []
accuraciesMeanItem = []
accuraciesMeanUser = []

for trainSet, testSet in yd.loadYahooDataset():
    trainSet.rating_scale=(1,13)
    
    for i in range(40,300,5):       
        kS.append(i)
        knnItem = KNNBaseline(sim_options = {'name': 'cosine', 'user_based': False}, k=i)
        knnUser = KNNBaseline(sim_options = {'name': 'cosine', 'user_based': True}, k=i)
        
        knnItem.fit(trainSet)
        knnUser.fit(trainSet)
        
        predictionsItem = knnItem.test(testSet)
        predictionsUser = knnUser.test(testSet)
        
        accuraciesBaselineItem.append(RecommenderMetrics.RMSE(predictionsItem))
        accuraciesBaselineUser.append(RecommenderMetrics.RMSE(predictionsUser))
        
        knnItem = KNNWithMeans(sim_options = {'name': 'cosine', 'user_based': False}, k=i)
        knnUser = KNNWithMeans(sim_options = {'name': 'cosine', 'user_based': True}, k=i)
        
        knnItem.fit(trainSet)
        knnUser.fit(trainSet)
        
        predictionsItem = knnItem.test(testSet)
        predictionsUser = knnUser.test(testSet)
        
        accuraciesMeanItem.append(RecommenderMetrics.RMSE(predictionsItem))
        accuraciesMeanUser.append(RecommenderMetrics.RMSE(predictionsUser))
        

plt.plot(kS, accuraciesMeanItem, label = "Item KNNWithMeans Collaborative Recommender")
plt.plot(kS, accuraciesMeanUser, label = "User KNNWithMeans  Collaborative Recommender")
plt.plot(kS, accuraciesBaselineItem, label = "Item KNNBaseline Collaborative Recommender")
plt.plot(kS, accuraciesBaselineUser, label = "User KNNBaseline Collaborative Recommender")

plt.ylabel('RMSE')
plt.xlabel('Broj suseda')

plt.title("Greška KNN sistema za preporuku ")
plt.legend(loc="upper left", bbox_to_anchor=(1, 1.025), shadow=True, ncol=1)
plt.show()     


