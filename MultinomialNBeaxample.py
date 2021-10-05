# Load Library
import numpy as np
from sklearn.naive_bayes import MultinomialNB
# Multinomial Naive Bayes is commonly used in text classification, the features here are the distribution of words frequency.
# For example, we want to use three words' - "Free", "Viagra", "Trial" - frequency to predict whether an email is a Spam or not

# Emails = [
# [Free Free Viagra Viagra!],
# [Viagra for Free.],
# [Free Trial Viagra ~],
# [Free Nail TRIAL],
# [Free Spa or Free Nail polish Trial !!!],
# [Top 10 Hiking Trial]])

Text_Freq = np.array([[2, 2, 0],
[1, 1, 0],
[1, 1, 1],
[1, 0, 1],
[1, 0, 2],
[0, 0, 1]])

X = Text_Freq
Y = np.array(['Spam', 'Spam', 'Spam', 'Not Spam', 'Not Spam', 'Not Spam'])
clf = MultinomialNB()
clf.fit(X, Y)

New_Email = 'Moonnight Trial'
print(clf.predict([[0,0,1]]))
['Not Spam']

New_Email2 = 'Free BitCoins'
print(clf.predict([[1,1,0]]))
# ['Not Spam']
# ['Spam']


