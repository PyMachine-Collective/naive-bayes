# Load Library
import numpy as np
from sklearn.naive_bayes import GaussianNB

smoke_density = np.array([[1.120], [2.256], [3.800], [0.768], [0.965], [0.362]])
X = smoke_density
Y = np.array(['FIRE', 'FIRE', 'FIRE', 'NOT FIRE', 'NOT FIRE', 'NOT FIRE'])
clf = GaussianNB()
clf.fit(X, Y)

print(clf.predict([[0.1]]))
print(clf.predict([[2.0]]))

# ['NOT FIRE']
# ['FIRE']
