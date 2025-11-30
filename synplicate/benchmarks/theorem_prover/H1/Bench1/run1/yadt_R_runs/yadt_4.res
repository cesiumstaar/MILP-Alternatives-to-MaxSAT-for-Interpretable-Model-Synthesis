YaDT version 2.2.0 (c) S. Ruggieri 2002-2019

MAJORITY CLASS value: 0 (2 classes)
MAJORITY MISCLASSIFICATION on training: 47.619% of 84 cases. 

MISCLASSIFICATION on training: 1.19048% of 84 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	43	1	0
0	0	40	1

MISCLASSIFICATION on test file: 15.6805% of 338 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	155	49	0
0	4	130	1

DECISION TREE (size=11)

F1 <= 0.100000 : 0 (25.0/1.3)
F1 > 0.100000 :
|   F10 <= 2.400000 :
|   |   F1 <= 0.980000 :
|   |   |   F10 <= 1.400000 : 0 (5.0/1.2)
|   |   |   F10 > 1.400000 :
|   |   |   |   F1 <= 0.410000 : 0 (3.0/1.1)
|   |   |   |   F1 > 0.410000 : 1 (6.0/1.2)
|   |   F1 > 0.980000 : 0 (10.0/1.3)
|   F10 > 2.400000 : 1 (35.0/2.6)
