YaDT version 2.2.0 (c) S. Ruggieri 2002-2019

MAJORITY CLASS value: 1 (2 classes)
MAJORITY MISCLASSIFICATION on training: 2.27273% of 132 cases. 

MISCLASSIFICATION on training: 0% of 132 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	3	0	0
0	0	129	1

MISCLASSIFICATION on test file: 9.24479% of 768 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	40	43	0
0	28	657	1

DECISION TREE (size=5)

age <= 30 :
|   age <= 29 : 0 (3.0/1.1)
|   age > 29 : 1 (5.0/1.2)
age > 30 : 1 (124.0/1.4)
