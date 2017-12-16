import numpy as np
from sklearn.neighbors import KNeighborsClassifier as knc
from sklearn.metrics import accuracy_score
from time import time

X=np.array([[2,5],[3,6],[1,7],[1,2],[4,3],[6,8],[7,3],[6,1],[8,7],[9,3]]);
Y=np.array([1,1,1,1,1,2,2,2,2,2]);

startTime=time()
clf=knc(n_neighbors=3)
clf.fit(X,Y)
pred=clf.predict([[0,1]]);
print(pred);

testX=np.array([[1,9],[3,1],[4,7],[6,5],[5,5],[7,9]]);
testY=np.array([1,1,1,2,2,2]);

pred=clf.predict(testX);
print("Accuracy ",accuracy_score(testY,pred)*100);

print("Time ",round(time()-startTime,3),"sec");