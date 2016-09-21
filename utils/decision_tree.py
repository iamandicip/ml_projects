## Load Data
data = pandas.read_csv('titanic_data.csv', sep=',')

## Define outcome, drop non used features and generate a binary variable for Sex:
outcomes = data['Survived']
data = data.drop(['Survived','Name','Ticket','Cabin','Embarked','PassengerId'], axis = 1)
data['Sex'] = data['Sex'].apply(lambda x: 1. if x == 'female' else 0.)

## Split the data into train/test sets: (train data is used to make the model learn from data and test data is used to estimate how well model generalized)
from sklearn.cross_validation import StratifiedShuffleSplit
sss = StratifiedShuffleSplit(outcomes, 1, test_size=0.3, random_state=450)
for train_index, test_index in sss:
    X_train = data.iloc[train_index]
    y_train = outcomes.iloc[train_index]
    X_test = data.iloc[test_index]
    y_test = outcomes.iloc[test_index]

## Define Decision Tree to use: (give it a try and change parameters to see the different trees built)
from sklearn import tree
clf = tree.DecisionTreeClassifier(max_features=3,max_depth=2)

## Use train data to train the model:
clf = clf.fit(X_train, y_train)

# Generate predictions over test set:
predictions = clf.predict(X_test)

# Accuracy results over test set:
from sklearn.metrics import accuracy_score
print "Accuracy Score:", accuracy_score(y_test, predictions)

## Represent Generated Tree:
from sklearn.externals.six import StringIO
from IPython.display import Image
import pydot
print data.columns
dot_data = StringIO()
tree.export_graphviz(clf, out_file=dot_data,
                         feature_names=data.columns,
                         class_names=['Perished','Survived'],
                         filled=True, rounded=True,
                         proportion = True,
                         special_characters=True)
graph = pydot.graph_from_dot_data(dot_data.getvalue())
Image(graph.create_png())
