# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 04:49:44 2018

@author: Srinivas
"""

# Text Classification

# Importing the libraries
import numpy as np
import re
import pickle
import nltk
from nltk.corpus import stopwords
from sklearn.datasets import load_files
nltk.download('stopwords')

# Importing the datasets
reviews = load_files('C:/Users/Srinivas/Desktop/Practice/AV/Movie Review Data/txt_sentoken/')
X,y = reviews.data,reviews.target

# To load file having large data load_files will take take time, so we use pickle to improve the performance

# Storing as pickle files
with open('X.pickle','wb') as f:  # wb = write in byte format
    pickle.dump(X,f)
with open('y.pickle','wb') as f:
    pickle.dump(y,f)
    
# Unpickling the dataset
with open('X.pickle','rb') as f:
    X = pickle.load(f)
    
with open('y.pickle','rb') as f:
    y = pickle.load(f)    
    
# Creating the corpus which mean reading the documents
corpus = []
for i in range(0,len(X)):
    review = re.sub(r'\W' , ' ',str(X[i])  )# Substituting all non-word characters with space
    review = review.lower()
    review = re.sub(r'\s+[a-z]\s+',' ',review )# Removing single characters which is having space before and after by replacing with space
    review = re.sub(r'^[a-z]\s+',' ',review) # Removing Single characters at the starting
    review = re.sub(r'\s+',' ',review) # Removing the Extra spaces we have created with single space
    corpus.append(review)

'''
# Lets create Bag Of Words Model

from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(max_features=2000,min_df=3,max_df=0.6,stop_words= stopwords.words('english'))
 # Here we are selecting max words=2000,minimum doc frequency =3 and max doc feq=60 % to delete the the repeated words from all doc's
 
X = vectorizer.fit_transform(corpus).toarray()   

'''
# Creating Tfidf Vectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(max_features=2000,min_df=3,max_df=0.6,stop_words= stopwords.words('english'))
 # Here we are selecting max words=2000,minimum doc frequency =3 and max doc feq=60 % to delete the the repeated words from all doc's
 
X = vectorizer.fit_transform(corpus).toarray()  


     
# Now we will convert Bag of Words Model to TF-IDF Model
from sklearn.feature_extraction.text import TfidfTransformer
transformer = TfidfTransformer()
X = transformer.fit_transform(X).toarray()

# Now splitting dataset into Train and Test

from sklearn.model_selection import train_test_split
text_train,text_test,sent_train,sent_test= train_test_split(X,y,test_size=0.2,random_state=0)


# Implementing Logistic Regression

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(text_train,sent_train)

sent_pred = classifier.predict(text_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(sent_test,sent_pred)


# Pickling the classifier to import directly to another analysis
with open('classifier.pickle','wb') as f:
    pickle.dump(classifier,f)
    
# Pickling the vectorizer
with open('tfidfmodel.pickle','wb') as f:
    pickle.dump(vectorizer,f)    
    
# Unpickling the classifier and vectorizer
with open('classifier.pickle','rb') as f:
    clf = pickle.load(f)
with open('tfidfmodel.pickle','rb') as f:
    tfidf = pickle.load(f)

sample =     ["You are a nice person man, have a good life"]
sample = tfidf.transform(sample).toarray()
print(clf.predict(sample)) # We got the output as polarity of 1 which means it is a positive sentence.
    
    