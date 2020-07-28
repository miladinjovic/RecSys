# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 17:30:54 2020

@author: Jovic
"""

from YahooDataset import YahooDataset
import matplotlib.pyplot as plt
from collections import defaultdict


import random
import numpy as np

def autolabel(rects):
    
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width()/2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


np.random.seed(0)
random.seed(0)


yd = YahooDataset()
demographicData = yd.loadDemographicsData()
demographicStats = demographicData.groupby('gender').count().reset_index().rename(columns={'userId':'count'})

trainSet = yd.loadYahooTrainDataset()
trainSet.rating_scale = (1, 13)

numUsers = 0

for userId in range(trainSet.n_users):
    mostLiked = 0
    for iid, rating in trainSet.ur[userId]:
        if rating > 9:
            mostLiked += 1
            
    if mostLiked/len(trainSet.ur[userId])>=0.5:
        numUsers += 1

print("Procenat korisnika sa ocenom većom od 9 za više od polovine odgledanih filmova: %.2f" % (numUsers/trainSet.n_users*100))

trainSet = yd.loadFullSet().build_full_trainset()
ratings = defaultdict(int)
for uid, iid, rating in trainSet.all_ratings():
    ratings[rating] += 1

ratings = dict(sorted(ratings.items()))

width = 0.8 
labels = list(range(1, len(ratings.keys())+1))
values = list(ratings.values())
_, ax = plt.subplots()
rects1 = ax.bar(labels, ratings.values(), width)
ax.set_ylabel('Učestalost')
ax.set_xlabel('Rejting')
ax.set_title('Raspodela rejtinga u apsolutnim brojevima')
ax.set_xticks(labels)
autolabel(rects1)


plt.show()
plt.close()

x = values[-5:]
x.append(sum(values[:8]))
explode = [0 for _ in range(len(x)-1)]
explode.append(0.5)
y = labels[-5:]
y.append("Ostali rejtinzi")
plt.pie(x,labels=y,autopct='%1.1f%%', explode = explode)
plt.title("Raspodela rejtinga u relativnim brojevima")


ax = demographicStats.plot.bar(x='gender', y='count', rot=0, legend=False, title="Raspodela korisnika prema polu")
ax.set_xlabel("Pol")
ax.set_ylabel("Broj korisnika")
autolabel(ax.patches)

otherStats = yd.loadYahooPandasFullDataFrame().merge(demographicData, left_on="userId", right_on="userId").groupby("gender", as_index=False).agg({"rating":"count"})
ax = otherStats.plot.bar(x='gender', y='rating', rot=0, legend=False, title="Raspodela rejtinga prema polu")
ax.set_xlabel("Pol")
ax.set_ylabel("Broj korisnika")
autolabel(ax.patches)