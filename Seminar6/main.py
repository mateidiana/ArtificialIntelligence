import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import sklearn as sk
#
#
df = pd.read_csv("faithful.csv")
data=np.array(df)
#
print(df)

from sklearn.cluster import KMeans

sse = []
for k in range(1, 10):
    kmeans = KMeans(n_clusters=k, random_state=42, init='random', n_init=10, max_iter=10)
    kmeans.fit(data)
    sse.append(kmeans.inertia_)

f, ax = plt.subplots(1,1,figsize=(15,9))
plt.plot(range(1, 10), sse)
plt.xticks(range(1, 10))
ax.annotate('Optimal number of clusters', xy=(2.05,9000))
plt.xlabel("Number of Clusters")
plt.ylabel("SSE")
plt.title('SSE for different number of clusters', fontsize = 20, c='black')
plt.show()