
from sklearn.cluster import KMeans as kmc
import numpy as np
import matplotlib.pyplot as plt


X=np.array([[1, 2], [1, 4], [1, 0],[4, 2], [4, 4], [4, 0]])

for ele in X:
    print(ele[0],ele[1])
    plt.scatter(ele[0],ele[1])
plt.show()

clt=kmc(n_clusters=2,random_state=0)
clt.fit(X)
print(clt.labels_)
print(clt.cluster_centers_)
print(clt.predict([[1.5,3],[1,3.5]]))

