import pandas as pd
from multiclass_svm import MultiClassSVM
from sklearn.model_selection import train_test_split


vect = pd.read_csv('vector.csv')
vect = vect.values

classes= pd.read_csv('dataset.csv')['coarse'].values
x_train, x_test, y_train, y_test = train_test_split(vect, classes)

clf = MultiClassSVM(1, 0.001, 0.001)
clf.train(x_train, y_train)

print(clf.score(x_test, y_test))