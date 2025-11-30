YaDT version 2.2.0 (c) S. Ruggieri 2002-2019

MAJORITY CLASS value: 1 (2 classes)
MAJORITY MISCLASSIFICATION on training: 47.2222% of 36 cases. 

MISCLASSIFICATION on training: 5.55556% of 36 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	15	2	0
0	0	19	1

MISCLASSIFICATION on test file: 30.4147% of 217 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	62	63	0
0	3	89	1

DECISION TREE (size=7)

F10 <= 2.900000 :
|   F1 <= 0.710000 : 0 (10.0/1.3)
|   F1 > 0.710000 :
|   |   F1 <= 0.990000 : 1 (6.0/2.3)
|   |   F1 > 0.990000 : 0 (5.0/1.2)
F10 > 2.900000 : 1 (15.0/2.5)
