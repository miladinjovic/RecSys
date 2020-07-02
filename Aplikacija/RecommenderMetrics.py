import itertools

from surprise import accuracy
from collections import defaultdict
import math

class RecommenderMetrics:
    
    def RRSE(predictions, globalMean, verbose = True):
        above = 0
        below = 0
        
        for _, _, actualRating, estimatedRating, _ in predictions:
            above += (estimatedRating - actualRating) * (estimatedRating - actualRating)
            below += (actualRating - globalMean) * (actualRating - globalMean)
        
        rrse = math.sqrt(above / below)
        if verbose :
            print("RRSE:", rrse)
        
        return rrse
    
    def MAE(predictions, verbose=True):
        return accuracy.mae(predictions, verbose)
    
    def RMSE(predictions, verbose=True):
        return accuracy.rmse(predictions, verbose)
    
    def GetTopNForUser(predictions, n=10, minimumRating=8.0):
        
        topN = []
        
        for userID, movieID, actualRating, estimatedRating, _ in predictions:
            if(estimatedRating >= minimumRating):
                topN.append( (int(movieID), estimatedRating) )
        
        topN.sort(key=lambda x: x[1], reverse = True)
        
        return  topN[:n]
    
    
    def HitRate(topNPredicted, leftOutPredictions):
        hits = 0
        total = 0

        for leftOut in leftOutPredictions:
            userID = leftOut[0]
            leftOutMovieID = leftOut[1]
            hit = False
            for movieID, predictedRating in topNPredicted[int(userID)]:
                if (int(leftOutMovieID) == int(movieID)):
                    hit = True
                    break
            if (hit) :
                hits += 1

            total += 1

        return hits/total


    def CumulativeHitRate(topNPredicted, leftOutPredictions, ratingCutoff=8.0):
        hits = 0
        total = 0

        for userID, leftOutMovieID, actualRating, estimatedRating, _ in leftOutPredictions:
            if (actualRating >= ratingCutoff):
                hit = False
                for movieID, predictedRating in topNPredicted[int(userID)]:
                    if (int(leftOutMovieID) == movieID):
                        hit = True
                        break
                if (hit) :
                    hits += 1

                total += 1
                
        return hits/total
    
    def RatingHitRate(topNPredicted, leftOutPredictions):
        hits = defaultdict(float)
        total = defaultdict(float)

        for userID, leftOutMovieID, actualRating, estimatedRating, _ in leftOutPredictions:
            hit = False
            for movieID, predictedRating in topNPredicted[int(userID)]:
                if (int(leftOutMovieID) == movieID):
                    hit = True
                    break
            if (hit) :
                hits[actualRating] += 1

            total[actualRating] += 1

        for rating in sorted(hits.keys()):
            print (rating, hits[rating] / total[rating])
            
            
    def AverageReciprocalHitRank(topNPredicted, leftOutPredictions):
        summation = 0
        total = 0
        
        for userID, leftOutMovieID, actualRating, estimatedRating, _ in leftOutPredictions:
            hitRank = 0
            rank = 0
            for movieID, predictedRating in topNPredicted[int(userID)]:
                rank = rank + 1
                if (int(leftOutMovieID) == movieID):
                    hitRank = rank
                    break
            if (hitRank > 0) :
                summation += 1.0 / hitRank

            total += 1

        return summation / total
    
    def UserCoverage(topNPredicted, numUsers, ratingThreshold=8.0):
        hits = 0
        for userID in topNPredicted.keys():
            hit = False
            for movieID, predictedRating in topNPredicted[userID]:
                if (predictedRating >= ratingThreshold):
                    hit = True
                    break
            if (hit):
                hits += 1

        return hits / numUsers
    
    
    def Diversity(topNPredicted, simsAlgo):
        n = 0
        total = 0
        simsMatrix = simsAlgo.compute_similarities()
        for userID in topNPredicted.keys():
            pairs = itertools.combinations(topNPredicted[userID], 2)
            for pair in pairs:
                movie1 = pair[0][0]
                movie2 = pair[1][0]
                innerID1 = simsAlgo.trainset.to_inner_iid(str(movie1))
                innerID2 = simsAlgo.trainset.to_inner_iid(str(movie2))
                similarity = simsMatrix[innerID1][innerID2]
                total += similarity
                n += 1

        S = total / n
        return (1-S)
    
    
    def Novelty(topNPredicted, rankings):
        n = 0
        total = 0
        for userID in topNPredicted.keys():
            for rating in topNPredicted[userID]:
                movieID = rating[0]
                rank = rankings[movieID]
                if(rank > 0):
                    total += rank
                    n += 1
                else:
                    continue
        
        return total / n