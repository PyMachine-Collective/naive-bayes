# Load Library
import numpy as np
from sklearn.naive_bayes import BernoulliNB

# Bernoulli Naive Bayes is commonly used in text classification too, the features here are the distribution of words' presence regardless it's frequency.
# Let's use the same email list as the ./MultinomialNBexample.py

Text_Pres = np.array([[1, 1, 0],
       [1, 1, 0],
       [1, 1, 1],
       [1, 0, 1],
       [1, 0, 1],
       [0, 0, 1]])
       
X = Text_Pres
Y = np.array(['Spam', 'Spam', 'Spam', 'Not Spam', 'Not Spam', 'Not Spam'])
clf = BernoulliNB()
clf.fit(X, Y)

New_Email = 'Moonnight Trial'
print(clf.predict([[0,0,1]]))
['Not Spam']

New_Email2 = 'Free BitCoins'
print(clf.predict([[1,1,0]]))
['Spam']


# ['Not Spam']
# ['Spam']

