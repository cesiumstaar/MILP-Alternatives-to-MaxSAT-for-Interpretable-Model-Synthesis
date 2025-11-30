YaDT version 2.2.0 (c) S. Ruggieri 2002-2019

MAJORITY CLASS value: 1 (2 classes)
MAJORITY MISCLASSIFICATION on training: 43% of 100 cases. 

MISCLASSIFICATION on training: 1% of 100 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	42	1	0
0	0	57	1

MISCLASSIFICATION on test file: 19.1892% of 370 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	127	69	0
0	2	172	1

DECISION TREE (size=13)

F1 <= -0.290000 : 0 (21.0/1.3)
F1 > -0.290000 :
|   F10 <= 1.600000 : 0 (13.0/1.3)
|   F10 > 1.600000 :
|   |   F1 <= 0.000000 :
|   |   |   F10 <= 4.300000 : 0 (4.0/1.2)
|   |   |   F10 > 4.300000 : 1 (4.0/2.2)
|   |   F1 > 0.000000 :
|   |   |   F10 <= 2.600000 :
|   |   |   |   F1 <= 1.200000 : 1 (5.0/1.2)
|   |   |   |   F1 > 1.200000 : 0 (4.0/1.2)
|   |   |   F10 > 2.600000 : 1 (49.0/1.4)
