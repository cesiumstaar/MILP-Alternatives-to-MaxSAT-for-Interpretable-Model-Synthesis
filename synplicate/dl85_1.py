"""
==========================================
DL8.5 classifier : native iterative search
==========================================
Iterative search is the idea that the algorithm starts with finding an optimal 
shallow tree, and then uses the quality of this tree to bound the quality of 
deeper trees. This class shows how to perform this type of search by using 'iterative'
option of the DL85Classifier class. In this case, an implementation in C++ is used.
A second implementation is shown in the plot_classifier_iterative_python.py class.
"""
import sys
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time
from dl85 import DL85Classifier

expt_dir = sys.argv[1]
xlimit = int(sys.argv[2])
#print (xlimit)
train_name = f"{expt_dir}/erm_syn_samples.dl85.txt"
test_name = f"{expt_dir}/dt_samples.dl85.txt"
#print (train_name)
#print (test_name)
dataset = np.genfromtxt(train_name, delimiter=' ')
#print (dataset)
Xtrain,ytrain = dataset[:, 0:xlimit], dataset[:, xlimit]
#print (Xtrain)
#y = dataset[:, 0]
#print ("printing Y")
#print (ytrain)
Xtrain = Xtrain.astype('int32')
ytrain = ytrain.astype('int32')

test_dataset = np.genfromtxt(test_name, delimiter=' ')
Xtest,ytest = test_dataset[:, 0:xlimit], test_dataset[:, xlimit]
Xtest = Xtest.astype('int32')
ytest = ytest.astype('int32')
#print (Xtest)
#print (ytest)
#Xtest = dataset[:, 1:]
#ytest = dataset[:, 0]

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#print ("### Shallow Classifer : Max depth 3, Min Sup 5 ")
#clf = DL85Classifier(max_depth=3, min_sup=5)
#start = time.perf_counter()
#print("Model building...")
#clf.fit(Xtrain, ytrain)
#duration = time.perf_counter() - start
#print("Model built. Duration of building =", round(duration, 4))
#y_pred = clf.predict(Xtest)
#print("Confusion Matrix below")
#print(confusion_matrix(ytest, y_pred))
#print("max depth = 3" )
#print("Accuracy DL8.5 on training set =", round(clf.accuracy_, 4))
#print("Accuracy DL8.5 on test set =", round(accuracy_score(ytest, y_pred), 4))



print("##########################################################################\n"
      "#            DL8.5 default classifier using iterative search  with different max depths #\n"
      "##########################################################################")

print("max depth = 3" )
clf = DL85Classifier(max_depth=5, iterative=True, time_limit=600, print_output=True)
start = time.perf_counter()
print("Model building...")
clf.fit(Xtrain, ytrain)
duration = time.perf_counter() - start
print("Model built. Duration of building =", round(duration, 4))
y_pred = clf.predict(Xtest)
print("Confusion Matrix below")
print(confusion_matrix(ytest, y_pred))
print("Accuracy DL8.5 on training set =", round(clf.accuracy_, 4))
print("Accuracy DL8.5 on test set =", round(accuracy_score(ytest, y_pred), 4))
#print ("----------------------------------------------------------------------")
#clf = DL85Classifier(max_depth=5, iterative=True, time_limit=600)
#start = time.perf_counter()
#print("Model building...")
#clf.fit(Xtrain, ytrain)
#duration = time.perf_counter() - start
#print("Model built. Duration of building =", round(duration, 4))
#y_pred = clf.predict(Xtest)
#print("Confusion Matrix below")
#print(confusion_matrix(ytest, y_pred))
#print("max depth = 5" )
#print("Accuracy DL8.5 on training set =", round(clf.accuracy_, 4))
#print("Accuracy DL8.5 on test set =", round(accuracy_score(ytest, y_pred), 4))

#print("###########################################################################\n"
#      "#      DL8.5 default classifier using python-based iterative search       #\n"
#      "###########################################################################")
#start = time.perf_counter()
#error = 0  # default max error value expressing no bound
#clf = None
#remaining_time = 600
#for i in range(1, 5):  # max depth = 2
#    clf = DL85Classifier(max_depth=i, max_error=error, time_limit=remaining_time)
#    clf.fit(Xtrain, ytrain)
#    error = clf.error_
#    remaining_time -= clf.runtime_
#duration = time.perf_counter() - start
#print("Model built. Duration of building =", round(duration, 4))
#print(f"Max depth =  {i}")
#y_pred = clf.predict(Xtest)
#print("Confusion Matrix below")
#print(confusion_matrix(ytest, y_pred))
#print("Accuracy DL8.5 on training set =", round(clf.accuracy_, 4))
#print("Accuracy DL8.5 on test set =", round(accuracy_score(ytest, y_pred), 4))


#print ("----------------------------------------------------------------------")
#print ("#### 100\% classifier ")
#maxdepth = 1
#while True:
#    clf = DL85Classifier(max_depth=maxdepth , min_sup=2,max_error=1)
#    clf.fit(Xtrain, ytrain)
#    if  clf.error_ == 0:
#        break
#    maxdepth += 1
#y_pred = clf.predict(Xtest)
#print (f"max depth = {maxdepth}")
#print("Accuracy DL8.5 on training set =", round(clf.accuracy_, 4))
#print("Accuracy DL8.5 on test set =", round(accuracy_score(ytest, y_pred), 4))
