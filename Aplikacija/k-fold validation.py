# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 09:50:03 2020

@author: Ja
"""

from YahooDataset import YahooDataset
from RecommenderMetrics import RecommenderMetrics
from surprise.model_selection import cross_validate
from surprise.model_selection import GridSearchCV
from surprise import KNNBasic, KNNWithMeans, KNNWithZScore, KNNBaseline
from surprise import SVD, SVDpp, NMF
from surprise import SlopeOne, CoClustering
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold


import random
import numpy as np

np.random.seed(0)
random.seed(0)

yd = YahooDataset()

data = yd.loadYahooPandasFullDataFrame()
data = yd.loadFromPandas(data)

# https://bmanohar16.github.io/blog/recsys-evaluation-in-surprise

# k-NN Based Algorithms

knnbasic_cv = cross_validate(KNNBasic(), data, cv=5, verbose=False)
knnmeans_cv = cross_validate(KNNWithMeans(), data, cv=5, verbose=False)
knnz_cv = cross_validate(KNNWithZScore(), data, cv=5,  verbose=False)
knnbaseline_cv = cross_validate(KNNBaseline(), data, cv=5, verbose=False)

# Matrix Factorization Based Algorithms

svd_cv = cross_validate(SVD(), data, cv=5, verbose=False)
svdpp_cv = cross_validate(SVDpp(), data, cv=5, verbose=False)
nmf_cv = cross_validate(NMF(), data, cv=5,  verbose=False)

# Other Collaborative Filtering Algorithms

slope_cv = cross_validate(SlopeOne(), data, cv=5,  verbose=False)
coclus_cv = cross_validate(CoClustering(), data, cv=5, verbose=False)

print('Algorithm\t RMSE\t\t MAE')
print()
print('KNN Basic', '\t', round(knnbasic_cv['test_rmse'].mean(), 4), '\t', round(knnbasic_cv['test_mae'].mean(), 4))
print('KNN Means', '\t', round(knnmeans_cv['test_rmse'].mean(), 4), '\t', round(knnmeans_cv['test_mae'].mean(), 4))
print('KNN ZScore', '\t', round(knnz_cv['test_rmse'].mean(), 4), '\t', round(knnz_cv['test_mae'].mean(), 4))
print('KNN Baseline', '\t', round(knnbaseline_cv['test_rmse'].mean(), 4), '\t', round(knnbaseline_cv['test_mae'].mean(), 4))
print()
print('SVD', '\t\t', round(svd_cv['test_rmse'].mean(), 4), '\t', round(svd_cv['test_mae'].mean(), 4))
print('SVDpp', '\t\t', round(svdpp_cv['test_rmse'].mean(), 4), '\t', round(svdpp_cv['test_mae'].mean(), 4))
print('NMF', '\t\t', round(nmf_cv['test_rmse'].mean(), 4), '\t', round(nmf_cv['test_mae'].mean(), 4))
print()
print('SlopeOne', '\t', round(slope_cv['test_rmse'].mean(), 4), '\t', round(slope_cv['test_mae'].mean(), 4))
print('CoClustering', '\t', round(coclus_cv['test_rmse'].mean(), 4), '\t', round(coclus_cv['test_mae'].mean(), 4))
print()


x_algo = ['KNN Basic', 'KNN Means', 'KNN ZScore', 'KNN Baseline', 'SVD', 'SVDpp', 'NMF', 'SlopeOne', 'CoClustering']
all_algos_cv = [knnbasic_cv, knnmeans_cv, knnz_cv, knnbaseline_cv, svd_cv, svdpp_cv, nmf_cv, slope_cv, coclus_cv]

rmse_cv = [round(res['test_rmse'].mean(), 4) for res in all_algos_cv]
mae_cv = [round(res['test_mae'].mean(), 4) for res in all_algos_cv]

plt.figure(figsize=(20,5))

plt.subplot(1, 2, 1)
plt.title('Poređenje algoritama (RMSE metrika)', loc='center', fontsize=15)
plt.plot(x_algo, rmse_cv, label='RMSE', color='darkgreen', marker='o')
plt.xlabel('Algoritmi', fontsize=15)
plt.ylabel('RMSE', fontsize=15)
plt.legend()
plt.xticks(rotation=90)
plt.grid(ls='dashed')

plt.subplot(1, 2, 2)
plt.title('Poređenje algoritama (MAE metrika)', loc='center', fontsize=15)
plt.plot(x_algo, mae_cv, label='MAE', color='navy', marker='o')
plt.xlabel('Algoritmi', fontsize=15)
plt.ylabel('MAE', fontsize=15)
plt.legend()
plt.grid(ls='dashed')
plt.xticks(rotation=90)

plt.show()


# Grid search params for 'best'

svd_param_grid = {'n_epochs': [20, 25], 
                  'lr_all': [0.003, 0.007, 0.009, 0.01],
                  'reg_all': [0.1, 0.4, 0.6],
                  'n_factors' : [20, 30]
                  }

svd_gs = GridSearchCV(SVD, svd_param_grid, measures=['rmse', 'mae'], cv=5)
svd_gs.fit(data)

print('SVD   - RMSE:', round(svd_gs.best_score['rmse'], 4), '; MAE:', round(svd_gs.best_score['mae'], 4))
print('RMSE =', svd_gs.best_params['rmse'])
print('MAE =', svd_gs.best_params['mae'])

knn_param_grid = {'k': [50, 60, 70, 80, 90, 100, 120, 150, 170, 200],
              'sim_options': {
                              'name': ['msd', 'cosine'],
                              'user_based': [False, True],
                              }
              }


knnbaseline_gs = GridSearchCV(KNNBaseline, knn_param_grid, measures=['rmse', 'mae'], cv=5)
knnbaseline_gs.fit(data)

print('KNNBaseline- RMSE:', round(knnbaseline_gs.best_score['rmse'], 4), '; MAE:', round(knnbaseline_gs.best_score['mae'], 4))
print('RMSE =', knnbaseline_gs.best_params['rmse'])
print('MAE =', knnbaseline_gs.best_params['mae'])


""" StratifiedKFold sampling """

# data = yd.loadYahooPandasFullDataFrame()

# svd = SVD(random_state = 0, reg_all=0.1, lr_all=0.003, n_factors=30, verbose=False)
# rmse = 0
# mae = 0
# skf = StratifiedKFold(n_splits=5)
# for train_index, test_index in skf.split(np.zeros(data["userId"].size), data["userId"]):
   #  data_train, data_test = data.iloc[train_index, :], data.iloc[test_index, :]
    # trainSet = yd.loadFromPandas(data_train).build_full_trainset()
    # testSet = yd.loadFromPandas(data_test).build_full_trainset().build_testset()
    # svd.fit(trainSet)
    # predictions = svd.test(testSet)
    # rmse += RecommenderMetrics.RMSE(predictions, verbose=False)
    # mae += RecommenderMetrics.MAE(predictions, verbose=False)
    
# print("Average RMSE on 5-fold", rmse / 5)#
# print("Average MAE on 5-fold", mae / 5)
    

