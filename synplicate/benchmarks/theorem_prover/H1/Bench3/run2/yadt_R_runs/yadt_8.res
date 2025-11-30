YaDT version 2.2.0 (c) S. Ruggieri 2002-2019

MAJORITY CLASS value: 0 (2 classes)
MAJORITY MISCLASSIFICATION on training: 49.3243% of 148 cases. 

MISCLASSIFICATION on training: 2.02703% of 148 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	72	3	0
0	0	73	1

MISCLASSIFICATION on test file: 23.2283% of 508 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	211	110	0
0	8	179	1

DECISION TREE (size=9)

F1 <= 0.100000 : 0 (51.0/1.4)
F1 > 0.100000 :
|   F10 <= 2.400000 :
|   |   F10 <= 1.600000 : 0 (15.0/1.3)
|   |   F10 > 1.600000 :
|   |   |   F1 <= 1.210000 : 1 (9.0/2.4)
|   |   |   F1 > 1.210000 : 0 (6.0/1.2)
|   F10 > 2.400000 : 1 (67.0/3.8)
