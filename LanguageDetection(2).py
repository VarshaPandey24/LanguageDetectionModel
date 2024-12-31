# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
import re
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter("ignore")

data = pd.read_csv("Language Detection.csv")
print(data.head(10))

data["Language"].value_counts()

#seperating dependent and independent data

#here language name is independent variable
#text data is dependent variable

x=data["Text"]
y=data["Language"]

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

#PREPROCESSING THE DATA

#creating a list to store preprocessed data

data_list=[]

#iterating through all the text

for text in x:
   text = re.sub(r'[!@#$(),n"%^*?:;~`0-9]', ' ', text)
   text = re.sub(r'[[]]', ' ', text)
   text = text.lower()
   data_list.append(text)

#we also need to give numerical form to our inputs
#so using bag of words for this

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
x = cv.fit_transform(data_list).toarray()
x.shape

#now we have to split the data in test set and training sets

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20)

#creating the model using naive bayes algorithm

from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
model.fit(x_train, y_train)

#after training the model predicting it
y_pred = model.predict(x_test)

#now calculating the accuracy of the model

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
ac = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("Accuracy is :",ac)

plt.figure(figsize=(15,10))
sns.heatmap(cm, annot = True)
plt.show()

def predict(text):
     x = cv.transform([text]).toarray() # converting text to bag of words model (Vector)
     lang = model.predict(x) # predicting the language
     lang = le.inverse_transform(lang) # finding the language corresponding the the predicted value
     return lang[0] # printing the language

predict("Analytics Vidhya provides a community based knowledge portal for Analytics and Data Science professionals")

predict("Analytics Vidhya fournit un portail de connaissances basé sur la communauté pour les professionnels de l'analyse et de la science des données")

predict("fournit un portail")

predict("അനലിറ്റിക്സ്, ഡാറ്റാ സയൻസ് പ്രൊഫഷണലുകൾക്കായി കമ്മ്യൂണിറ്റി അധിഷ്ഠിത വിജ്ഞാന പോർട്ടൽ അനലിറ്റിക്സ് വിദ്യ നൽകുന്നു")

predict("это портал знаний на базе сообщества для профессионалов в области аналитики и данных")
