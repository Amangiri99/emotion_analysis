import pandas as pd
import numpy as np
import nltk
train_df = pd.read_csv("C:/Users/Aman/Desktop/Project_NLP/data/train.txt",sep = ';',names = ["sentence","emotion"])
test_df = pd.read_csv("C:/Users/Aman/Desktop/Project_NLP/data/test.txt",sep = ';',names = ["sentence","emotion"])
val_df = pd.read_csv("C:/Users/Aman/Desktop/Project_NLP/data/val.txt",sep = ';',names = ["sentence","emotion"])
nltk.download()
train_df.head()
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
snow = SnowballStemmer("english")
corpus_train = []
sw = stopwords.words("english")
print(len(train_df))
for i in range(0,len(train_df)):
    review = re.sub('[^a-zA-z]', ' ', train_df['sentence'][i])
    review = review.lower()
    review = review.split()
    
    review = [snow.stem(word) for word in review if word not in stopwords.words("english")]
    review = ' '.join(review)
    corpus_train.append(review)
corpus_test = []    
for i in range(0,len(test_df)):
    review = re.sub('[^a-zA-z]', ' ', test_df['sentence'][i])
    review = review.lower()
    review = review.split()
    
    review = [snow.stem(word) for word in review if word not in stopwords.words("english")]
    review = ' '.join(review)
    corpus_test.append(review)
    

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 3000)
X_train = cv.fit_transform(corpus_train).toarray()
X_test = cv.transform(corpus_test).toarray()

from sklearn import preprocessing
lb = preprocessing.LabelEncoder()
train_df['emotion'] = lb.fit_transform(train_df['emotion'])
y_train = train_df['emotion'].values
test_df['emotion'] = lb.transform(test_df['emotion'])
y_test = test_df['emotion'].values

print(lb.classes_)
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()
clf.fit(X_train,y_train)

from sklearn.metrics import confusion_matrix
pred = clf.predict(X_test)
con_matrix = confusion_matrix(y_test,pred)

from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test,pred)
print(acc)




import pickle
pickle.dump(cv, open('cv.pkl', 'wb'))
pickle.dump(clf, open("clf.pkl", "wb"))
pickle.dump(lb, open("lb.pkl", "wb"))


loaded_cv = pickle.load(open("cv.pkl", "rb"))
loaded_clf = pickle.load(open("clf.pkl", "rb"))
loaded_lb = pickle.load(open("lb.pkl", "rb"))


def new_review(new_review):
    
  new_review = new_review
  new_review = re.sub('[^a-zA-Z]', ' ', new_review)
  new_review = new_review.lower()
  new_review = new_review.split()
  snow = SnowballStemmer('english')
  sw = stopwords.words('english')
  new_review = [snow.stem(word) for word in new_review if not word in set(sw)]
  new_review = ' '.join(new_review)
  new_corpus = [new_review]
  new_X_test = loaded_cv.transform(new_corpus).toarray()
  new_y_pred = loaded_clf.predict(new_X_test)
        
  arr = loaded_lb.classes_
  arr[new_y_pred]
  return new_y_pred
my_text = "There is no one hard and fast definition for the term happiness. Happiness differs from person to person; different people have different perceptions and conceptions of being happy. Whatever that may be, Happiness is an essential feature of human life. Without it, life holds no meaning at all. It is not possible at all for a person to live their lives devoid of joy and Happiness."
pred = new_review(my_text)
arr = loaded_lb.classes_
#print(arr)
print(arr[pred])
