# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 11:17:38 2020

@author: Jovic
"""

import itertools
from collections import defaultdict

class bprRecommenderMetrics:
    
    def HitRate(topNPredicted, leftOutPredictions):
        hits = 0
        total = 0
        
        for user, row in enumerate(leftOutPredictions):
            if(len(row.indices) == 0):
                continue
            for rec in topNPredicted[user]:
                if(rec == row.indices[0]):
                    hits += 1
                    break
            total += 1
            
        return hits/total


    def CumulativeHitRate(topNPredicted, leftOutPredictions, ratingCutoff=8.0, df=None):
        hits = 0
        total = 0
        
        for user, row in enumerate(leftOutPredictions):
            if(len(row.indices) == 0):
                continue
            
            movieId = df["movieId"].cat.categories[row.indices[0]]
            userId = df["userId"].cat.categories[user]
            actualRating = df[(df["userId"] == userId) & (df["movieId"] == movieId)]["rating"].values[0]
            
            if(actualRating >= ratingCutoff):
                for rec in topNPredicted[user]:
                    if(rec == row.indices[0]):
                        hits += 1
                total += 1
                                
        return hits/total
    
    def RatingHitRate(topNPredicted, leftOutPredictions, df=None):
        hits = defaultdict(float)
        total = defaultdict(float)
        
        for user, row in enumerate(leftOutPredictions):
            if(len(row.indices) == 0):
                continue
            
            movieId = df["movieId"].cat.categories[row.indices[0]]
            userId = df["userId"].cat.categories[user]
            actualRating = df[(df["userId"] == userId) & (df["movieId"] == movieId)]["rating"].values[0]
            
            for rec in topNPredicted[user]:
                if(rec == row.indices[0] ):
                    hits[actualRating] += 1
                    break
                
            total[actualRating] += 1
    
        for rating in sorted(hits.keys()):
            print (rating, hits[rating] / total[rating])
            
            
    def AverageReciprocalHitRank(topNPredicted, leftOutPredictions):
        summation = 0
        total = 0
        
        for user, row in enumerate(leftOutPredictions):
            hitRank = 0
            rank = 0
            if(len(row.indices) == 0):
                continue
            for rec in topNPredicted[user]:
                rank = rank + 1
                if (rec == row.indices[0]):
                    hitRank = rank
                    break
            if (hitRank > 0) :
                summation += 1.0 / hitRank

            total += 1

        return summation / total
    
#    def UserCoverage(topNPredicted, numUsers, ratingThreshold=8.0):
#        hits = 0
#        for userID in topNPredicted.keys():
#            hit = False
#            for movieID, predictedRating in topNPredicted[userID]:
#                if (predictedRating >= ratingThreshold):
#                    hit = True
#                    break
#            if (hit):
#                hits += 1
#
#        return hits / numUsers
    
    
    def Diversity(topNPredicted, simsAlgo, df=None):
        n = 0
        total = 0
        simsMatrix = simsAlgo.compute_similarities()
        for user, row in enumerate(topNPredicted):
            pairs = itertools.combinations(topNPredicted[user], 2)
            for pair in pairs:
                movie1 = pair[0]
                movie2 = pair[1]
                movie1 = df["movieId"].cat.categories[movie1]
                movie2 = df["movieId"].cat.categories[movie2]
                innerID1 = simsAlgo.trainset.to_inner_iid(str(movie1))
                innerID2 = simsAlgo.trainset.to_inner_iid(str(movie2))
                similarity = simsMatrix[innerID1][innerID2] 
                total += similarity
                n += 1
                
        S = total / n
        return (1-S)
        
    def Novelty(topNPredicted, rankings, df=None):
        n = 0
        total = 0
        for user, row in enumerate(topNPredicted):
            for rec in topNPredicted[user]:
                movieId = df["movieId"].cat.categories[rec]
                rank = rankings[movieId]
                if(rank > 0):
                    total += rank
                    n += 1
                else:
                    continue
        
        return total / n
    