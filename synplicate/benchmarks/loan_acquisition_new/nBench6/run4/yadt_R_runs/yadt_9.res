YaDT version 2.2.0 (c) S. Ruggieri 2002-2019

MAJORITY CLASS value: 1 (2 classes)
MAJORITY MISCLASSIFICATION on training: 1.82927% of 164 cases. 

MISCLASSIFICATION on training: 0% of 164 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	3	0	0
0	0	161	1

MISCLASSIFICATION on test file: 8.49282% of 836 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	40	43	0
0	28	725	1

DECISION TREE (size=5)

age <= 31 :
|   age <= 29 : 0 (3.0/1.1)
|   age > 29 : 1 (10.0/1.3)
age > 31 : 1 (151.0/1.4)
