# -*- coding: utf-8 -*-
"""
Created on Sun Dec 27 06:36:11 2020

@author: Asus
"""
#packages
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
#import the dataset
messages = pd.read_csv("C:/Users/Asus/Pictures/SMSSpamCollection.txt",sep='\t',names=['label','message'])

#datacleaning and preprocessing
ps = PorterStemmer()
wordnet=WordNetLemmatizer()
corpus = []
for i in range(0,len(messages)):
    review = re.sub('[^a-zA-Z]', ' ', messages['message'][i])
    review = review.lower()
    review = review.split()
    review = [wordnet.lemmatize(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
    
#create a bag of words model
cv = CountVectorizer(max_features=5000)
X = cv.fit_transform(corpus).toarray()
 
y = pd.get_dummies(messages['label'])
y = y.iloc[:,1].values
#train and test split
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2,random_state = 0)
#model triraning
from sklearn.naive_bayes import MultinomialNB
spam_model = MultinomialNB().fit(X_train, y_train)
y_pred = spam_model.predict(X_test)

#accuracy of spam_model
from sklearn.metrics import confusion_matrix
confusion_m = confusion_matrix(y_test,y_pred)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)