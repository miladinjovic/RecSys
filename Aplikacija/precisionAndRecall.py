# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 09:10:08 2020

@author: Ja
"""

from collections import defaultdict
from surprise import Dataset

from surprise import KNNBasic, KNNWithMeans, KNNWithZScore, KNNBaseline
from surprise import SVD, SVDpp, NMF
from surprise import SlopeOne, CoClustering

from YahooDataset import YahooDataset
from surprise.model_selection import KFold
import matplotlib.pyplot as plt

import random
import numpy as np

np.random.seed(0)
random.seed(0)


# function from surprise 

def precision_recall_at_k(predictions, k=10, threshold=3.5):
    '''Return precision and recall at k metrics for each user.'''

    # First map the predictions to each user.
    user_est_true = defaultdict(list)
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    precisions = dict()
    recalls = dict()
    for uid, user_ratings in user_est_true.items():

        # Sort user ratings by estimated value
        user_ratings.sort(key=lambda x: x[0], reverse=True)

        # Number of relevant items
        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)

        # Number of recommended items in top k
        n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])

        # Number of relevant and recommended items in top k
        n_rel_and_rec_k = sum(((true_r >= threshold) and (est >= threshold))
                              for (est, true_r) in user_ratings[:k])

        # Precision@K: Proportion of recommended items that are relevant
        precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 0

        # Recall@K: Proportion of relevant items that are recommended
        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 0

    return precisions, recalls


yd = YahooDataset()
data = yd.loadYahooPandasFullDataFrame()
data = yd.loadFromPandas(data)
kf =KFold(n_splits=5)

precisionByAlgortihm = []
recallByAlgorithm = []

for algo in [KNNBasic(), KNNWithMeans(), KNNWithZScore(), KNNBaseline(), SVD(), SVDpp(), NMF(), SlopeOne(), CoClustering()]:
    p = 0
    r = 0
    for trainSet, testSet in kf.split(data):
        trainSet.rating_scale = (1, 13)
        algo.fit(trainSet)
        predictions = algo.test(testSet)
        precisions, recalls = precision_recall_at_k(predictions, k=10, threshold=9)
    
        p += sum(prec for prec in precisions.values()) / len(precisions)
        r += sum(rec for rec in recalls.values()) / len(recalls)
    
    precisionByAlgortihm.append(round(p/5, 4))
    recallByAlgorithm.append(round(r/5, 4))


names = ['KNN Basic', 'KNN Means', 'KNN ZScore', 'KNN Baseline', 'SVD', 'SVDpp', 'NMF', 'SlopeOne', 'CoClustering']

plt.figure(figsize=(20,5))

plt.subplot(1, 2, 1)
plt.title('Poređenje algoritama (Precision)', loc='center', fontsize=15)
plt.plot(names, precisionByAlgortihm, label='Precision', color='darkgreen', marker='o')
plt.xlabel('Algoritmi', fontsize=15)
plt.ylabel('Precision', fontsize=15)
plt.legend()
plt.xticks(rotation=90)
plt.grid(ls='dashed')

plt.subplot(1, 2, 2)
plt.title('Poređenje algoritama (Recall)', loc='center', fontsize=15)
plt.plot(names, recallByAlgorithm, label='Recall', color='navy', marker='o')
plt.xlabel('Algoritmi', fontsize=15)
plt.ylabel('Recall', fontsize=15)
plt.legend()
plt.grid(ls='dashed')
plt.xticks(rotation=90)

plt.show()

    
# algo = SVD(random_state = 0, reg_all=0.1, lr_all=0.003, n_factors=30, verbose=False)
# algo = KNNBaseline(sim_options = {'name': 'cosine', 'user_based': False}, k=150)

# for trainset, testset in yd.loadYahooDataset():
#     trainset.rating_scale = (1, 13)
#     algo.fit(trainset)
#     predictions = algo.test(testset)
#     precisions, recalls = precision_recall_at_k(predictions, k=10, threshold=9.0)

    # Precision and recall can then be averaged over all users
#     print(sum(prec for prec in precisions.values()) / len(precisions))
#     print(sum(rec for rec in recalls.values()) / len(recalls))
