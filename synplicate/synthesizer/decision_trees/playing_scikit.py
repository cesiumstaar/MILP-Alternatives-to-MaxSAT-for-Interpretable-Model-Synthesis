from sklearn import tree 
import graphviz
import os

X = [[0.9,0.4,0],[1.4,5.7,0],[0,0,1]]
Y = [0,2,1]



clf = tree.DecisionTreeClassifier()
clf = clf.fit(X,Y)

prediction = clf.predict(X)

count =0 

for i in range(len(Y)):
    if Y[i]!=prediction[i]:
        count +=1

print(count/len(Y))

dot_data = tree.export_graphviz(clf, out_file=None, feature_names=["A","B","C"], class_names=["X","Y","Z"])
graph = graphviz.Source(dot_data)
graph.render("iris")

os.system(f"dot -Tpng iris -o iris.png")