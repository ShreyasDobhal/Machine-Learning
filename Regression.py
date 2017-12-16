import numpy as np
from sklearn.linear_model import LinearRegression as lr
# This uses ordinary least squares
from time import time
import matplotlib.pyplot as plt

#X=np.array([[0],[1],[2],[3],[4]])
#Y=np.array([0,2,4,5,7])

X=np.array([[20,80],[20,100],[20,120],[40,80],[40,100],[40,120],[60,80],[60,100],[60,120]])
Y=np.array([100,150,200,200,250,300,300,350,400])

startTime=time()
reg=lr()
reg.fit(X,Y)

#plt.scatter(X,Y)
#plt.plot(X,reg.predict(X),linewidth=3,color='y')
#plt.xlabel("X -- > ")
#plt.ylabel("Y -- > ")
#plt.show()


print("Slope ",reg.coef_)
print("Intercept ",round(reg.intercept_,3))
#print("Prediction ",reg.predict(10))
print("R-sq Score ",reg.score(X,Y))
print("Time ",round(time()-startTime,3),"sec")