YaDT version 2.2.0 (c) S. Ruggieri 2002-2019

MAJORITY CLASS value: 1 (2 classes)
MAJORITY MISCLASSIFICATION on training: 2.02703% of 148 cases. 

MISCLASSIFICATION on training: 0% of 148 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	3	0	0
0	0	145	1

MISCLASSIFICATION on test file: 8.74384% of 812 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	40	43	0
0	28	701	1

DECISION TREE (size=5)

age <= 30 :
|   age <= 29 : 0 (3.0/1.1)
|   age > 29 : 1 (5.0/1.2)
age > 30 : 1 (140.0/1.4)
