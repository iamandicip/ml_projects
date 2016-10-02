#!/usr/bin/python

""" lecture and example code for decision tree unit """

import sys
from class_vis import prettyPicture, output_image
from prep_terrain_data import makeTerrainData

import matplotlib.pyplot as plt
import numpy as np
import pylab as pl
from classifyDT import classify

def classify(features_train, labels_train):

    ### your code goes here--should return a trained decision tree classifer
    from sklearn.tree import DecisionTreeClassifier

    clf = DecisionTreeClassifier(random_state = 42)
    clf.fit(features_train, labels_train)

    return clf

features_train, labels_train, features_test, labels_test = makeTerrainData()

### the classify() function in classifyDT is where the magic
### happens--fill in this function in the file 'classifyDT.py'!
clf = classify(features_train, labels_train)
acc = clf.score(features_test, labels_test)

#### grader code, do not modify below this line

prettyPicture(clf, features_test, labels_test)
output_image("test.png", "png", open("test.png", "rb").read())

import sys
from class_vis import prettyPicture
from prep_terrain_data import makeTerrainData

import matplotlib.pyplot as plt
import numpy as np
import pylab as pl

features_train, labels_train, features_test, labels_test = makeTerrainData()



########################## DECISION TREE #################################


### your code goes here--now create 2 decision tree classifiers,
### one with min_samples_split=2 and one with min_samples_split=50
### compute the accuracies on the testing data and store
### the accuracy numbers to acc_min_samples_split_2 and
### acc_min_samples_split_50, respectively
from sklearn.tree import DecisionTreeClassifier

clf_2 = DecisionTreeClassifier(min_samples_split=2, random_state = 42)
clf_2.fit(features_train, labels_train)
acc_min_samples_split_2 = clf_2.score(features_test, labels_test)

clf_50 = DecisionTreeClassifier(min_samples_split=50, random_state = 42)
clf_50.fit(features_train, labels_train)
acc_min_samples_split_50 = clf_50.score(features_test, labels_test)


def submitAccuracies():
  return {"acc_min_samples_split_2":round(acc_min_samples_split_2,3),
          "acc_min_samples_split_50":round(acc_min_samples_split_50,3)}
