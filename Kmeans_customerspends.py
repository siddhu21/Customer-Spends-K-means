# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 19:40:45 2020

@author: nsidd
"""

#Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Importing Dataset
data = pd.read_csv("customerspends.csv") 
sns.lmplot("Apparel","Beauty and Healthcare",data=data,fit_reg = False, size=4)

#Normalizing features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_data= scaler.fit_transform(data[["Apparel","Beauty and Healthcare"]])
scaled_data[0:5]

#Scaled SCatter Plot
sns.lmplot("Apparel","Beauty and Healthcare",data=data,fit_reg = False, size=5)


#Import KMeans
from sklearn.cluster import KMeans
clusters_new = KMeans(3, random_state= 0)
clusters_new.fit(scaled_data)
data['clusterid_new']=clusters_new.labels_


markers = ['+','^','.']
sns.lmplot("Apparel","Beauty and Healthcare",data=data, hue= 'clusterid_new',fit_reg= False, markers= markers, size=5)
clusters_new.cluster_centers_   

#Using Dendogram
cmap= sns.cubehelix_palette(as_cmap = True, rot=-0.3, light=1)
sns.clustermap(scaled_data,cmap=cmap,linewidths=0.2, figsize=(8,8))

#Using Elbow Method for No.of Clusters
from sklearn.cluster import KMeans
wcss = []
for i in range(1,21):
	kmeans = KMeans(n_clusters=i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
	kmeans.fit(scaled_data)
	wcss.append(kmeans.inertia_)
plt.plot(range(1,21),wcss)	
plt.title('Elbow Method')
plt.xlabel('No.of Clusters')
plt.ylabel('WCSS')
plt.show()


#Applying Kmeans with 3 Clusters
kmeans = KMeans(n_clusters= 3, init = 'k-means++', max_iter = 400, n_init = 10, random_state = 0)
y_kmeans= kmeans.fit_predict(scaled_data)

plt.scatter(scaled_data[y_kmeans == 0,0],scaled_data[y_kmeans == 0,1], s=70, c='red', label='FashionIsta')
plt.scatter(scaled_data[y_kmeans == 1,0],scaled_data[y_kmeans == 1,1], s=70, c='blue', label='BeautyObsession')
plt.scatter(scaled_data[y_kmeans == 2,0],scaled_data[y_kmeans == 2,1], s=70, c='green', label=' Both')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1], s=300, c='yellow', label=' Centroids')
plt.title("Customer Spends")
plt.xlabel("Apparel")
plt.ylabel("Beauty & Healthcare")
plt.legend()
plt.show()


#So by Elbow Method we have 3 clusters i.e., FashionISta,BeautyObsession, Both.

