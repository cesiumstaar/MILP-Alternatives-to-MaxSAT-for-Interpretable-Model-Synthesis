YaDT version 2.2.0 (c) S. Ruggieri 2002-2019

MAJORITY CLASS value: 1 (2 classes)
MAJORITY MISCLASSIFICATION on training: 3% of 100 cases. 

MISCLASSIFICATION on training: 0% of 100 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	3	0	0
0	0	97	1

MISCLASSIFICATION on test file: 7.57143% of 700 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	50	33	0
0	20	597	1

DECISION TREE (size=5)

age <= 34 :
|   age <= 29 : 0 (3.0/1.1)
|   age > 29 : 1 (6.0/1.2)
age > 34 : 1 (91.0/1.4)
