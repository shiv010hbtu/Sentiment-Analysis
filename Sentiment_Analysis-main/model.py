# Some Standard imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Retrieving the data from the dataset
data = pd.read_csv('data/dataset.csv')
data.head()


# Making the features and label 
x = data['review']
y = data['sentiment']
# x.shape
# y.shape

# Now word tokenization and stopword removal
from nltk.tokenize import sent_tokenize , word_tokenize , RegexpTokenizer

# Tokenizing each review using list comprehension
x = [word_tokenize(t) for t in x]


# StopWord Removal from the tokenized sentence
from nltk.corpus import stopwords
sw = stopwords.words('english')

# Adding some punctuation to the sw so as to remove them too
pun = ['.','`','~','!','@','#','$','%','^','&','*','(',')','-','_','=','+',',','<','>','.','/','?',"'",';',':','[',']']
sw = sw + pun

x = [[i for i in t if i not in sw and len(i)>0] for t in x]

for i in range(len(x)):
    for j in range(len(x[i])):
        x[i][j] = x[i][j].lower()
# for i in range(len(x)):
#     for j in range(len(x[i])):
#         x[i][j] = x[i][j].lower()

# We can also use RegexpTokenizer to do the same
tokenizer = RegexpTokenizer(r'\w+')
# for t in range(len(x)):
#     for i in range(len(str(t))):
#         if len(tokenizer.tokenize(x[t][i]))>0:
#             x[t][i] = ''.join(tokenizer.tokenize(x[t][i]))
#     x[t] = ' '.join(x[t])
x = [[''.join(tokenizer.tokenize(word)) for word in t if len(tokenizer.tokenize(word))>0] for t in x]

# Stemming 
from nltk.stem import PorterStemmer , LancasterStemmer
porter = PorterStemmer()
lancaster = LancasterStemmer()


# x = [ [porter.stem(word) for word in i] for i in x]
x = [ [lancaster.stem(word) for word in i] for i in x]


#  Make the sentences now as they are in word form
clean_data = [' '.join(t) for t in x]



# Now we have to do the vectorization of the sentences
from sklearn.feature_extraction.text import CountVectorizer , TfidfVectorizer

vectorizer = CountVectorizer(ngram_range=(1,3))
x = vectorizer.fit_transform(clean_data)



# Converting data of sentiment into numbers
y = y.apply( lambda x : 1 if x=='positive' else 0)

# Splitting our data into testing and training

from sklearn.model_selection import train_test_split

x_train , x_test , y_train , y_test = train_test_split(x,y,test_size=0.2)



# Apply the algorithm -> Logistic Regression
from sklearn.linear_model import LogisticRegression
# from 

model1 = LogisticRegression()



# Fitting the data into the model
from sklearn.ensemble import RandomForestClassifier

model2 = RandomForestClassifier()

# model1.fit(x_train,y_train)


model2.fit(x_train,y_train)

sc = model2.score(x_test,y_test)

print(sc)


