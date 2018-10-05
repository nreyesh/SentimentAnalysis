
### Text Analytics - Assigment 2 ###
### Author: Nicolás Reyes Huerta - ID:1700927 ###


### Library used
from numpy import *
import string
import sklearn
import nltk
from nltk.corpus.reader import ConllCorpusReader

import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics
import scipy.stats

from sklearn.metrics import make_scorer
from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import RandomizedSearchCV


### Reading and pre-procesing

## Reads wp2 Train Set

list_train = open("/home/nicor/Documents/aij-wikiner-en-wp2.txt").read().splitlines()
list_train = [st for st in list_train if st != '']
lst = []
for i in range(0,len(list_train)):
    lst.append(list_train[i].split())


lst_train = []
for i in range(0,len(lst)):
    lst2 = []
    for j in range(0,len(lst[i])):
        lst2.append(lst[i][j].split('|'))
    lst_train.append(lst2)

print('Train Set Readed')

## Reads the WikiGold Test Set

reader = ConllCorpusReader('/home/nicor/Documents','.conll',('words','pos'))
list_test = reader.tagged_sents('wikigold.txt')

lst_test = []
for i in range(0,len(list_test)):
    lst1 = []
    lst2 = []
    lst3 = []
    for j in range(0, len(list_test[i])):
        lst1.append(list_test[i][j][0])
    list2 = nltk.pos_tag(lst1)
    for j in range(0, len(list2)):
        lst3.append([list2[j][0], list2[j][1], list_test[i][j][1]])
    lst_test.append(lst3)

### Defines the Features to be obtained from every sentence

def word2features(sent, i):
    word = sent[i][0]
    postag = sent[i][1]

    features = {
        'word.lower()': word.lower(),               # returns a loweredcase word
         'word.isupper()': word.isupper(),           # checks if the word is upper case
        'word.istitle()': word.istitle(),           # checks if first letter of the word is upper case
        'word.isdigit()': word.isdigit(),           # checks if word is a digit
         'postag': postag,                           # obtains postag of the word
    }

    if i > 0:
        word1 = sent[i-1][0]
        postag1 = sent[i-1][1]
        features.update({
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:word.isdigit()': word.isdigit(),
            '-1:postag': postag1,
        })
    else:
        features['BOS'] = True

    if i < len(sent)-1:
        word1 = sent[i+1][0]
        postag1 = sent[i+1][1]
        features.update({
            '+1:word.lower()': word.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '-1:word.isdigit()': word.isdigit(),
            '+1:postag': postag1,
        })
    else:
        features['EOS'] = True
    return features


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    return [label for token, postag, label in sent]

def sent2tokens(sent):
    return [token for token, postag, label in sent]


### Extracting Features ###
# Train Set
x_train = [sent2features(s) for s in lst_train]
y_train = [sent2labels(s) for s in lst_train]

# Test Set
x_test = [sent2features(s) for s in lst_test]
y_test = [sent2labels(s) for s in lst_test]


### Training the Model ###
# Defines the model
crf = sklearn_crfsuite.CRF(
    algorithm='lbfgs',
    c1=0.1,
    c2=0.1,
    max_iterations=50,
    all_possible_transitions=True)

# trains the model
crf.fit(x_train, y_train)


### Testing the Model ###
# Removes unnecesary labels
labels = list(crf.classes_)
labels.remove('O')
labels.remove('B-MISC')
labels.remove('B-PER')
labels.remove('B-ORG')
labels.remove('B-LOC')
labels

# Makes prediction
y_pred = crf.predict(x_test)

# Accuracy obtained
metrics.flat_f1_score(y_test, y_pred,
                      average='weighted', labels=labels)

# Prints Presicion, Recall and F-score per entity 
sorted_labels = sorted(labels,key=lambda name: (name[1:], name[0]))

print(metrics.flat_classification_report(
    y_test, y_pred, labels=sorted_labels, digits=3))

### Tunning the Model ###

x_train = [sent2features(s) for s in lst_train[0:50000]]
y_train = [sent2labels(s) for s in lst_train[0:50000]]

# Defines the model
crf_opt = sklearn_crfsuite.CRF(
    		algorithm='lbfgs',
    		max_iterations=50,
   		all_possible_transitions=True)

# Generates a random number for C1 and C2 
params_space = {'c1': scipy.stats.expon(scale=0.5),
		'c2': scipy.stats.expon(scale=0.05)}

# Defines the metric which will be used to evaluate performance
f1_scorer = make_scorer(metrics.flat_f1_score,
                        average='weighted', labels=labels)

# Defines a 10 fold cros-validation method with 10 iteration each, to obtain different models for the C1 and C2 randomly generated
rs = RandomizedSearchCV(crf_opt, params_space,
                        cv=10,
                        verbose=1,
                        n_jobs=1,
                        n_iter=10,
                        scoring=f1_scorer)

# Trains several models with characteristic defined above
rs.fit(x_train, y_train)

# Prints the best model trained from the previous step
print('best params:', rs.best_params_)
print('best CV score:', rs.best_score_)

# Choose the C1 and C2 which produced the best model
crf_final = rs.best_estimator_

# Predict entities using the parameters which produce the best model
y_pred = crf_final.predict(x_test)

# Prints Presicion, Recall and F-score per entity 
print(metrics.flat_classification_report(
    y_test, y_pred, labels=labels, digits=3))

