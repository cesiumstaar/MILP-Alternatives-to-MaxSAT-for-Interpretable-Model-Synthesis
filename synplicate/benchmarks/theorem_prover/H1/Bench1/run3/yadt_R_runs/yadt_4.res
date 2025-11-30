YaDT version 2.2.0 (c) S. Ruggieri 2002-2019

MAJORITY CLASS value: 1 (2 classes)
MAJORITY MISCLASSIFICATION on training: 44.0476% of 84 cases. 

MISCLASSIFICATION on training: 1.19048% of 84 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	36	1	0
0	0	47	1

MISCLASSIFICATION on test file: 17.4174% of 333 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	119	57	0
0	1	156	1

DECISION TREE (size=13)

F10 <= 1.600000 : 0 (18.0/1.3)
F10 > 1.600000 :
|   F1 <= 0.000000 :
|   |   F1 <= -0.290000 : 0 (10.0/1.3)
|   |   F1 > -0.290000 :
|   |   |   F10 <= 4.300000 : 0 (4.0/1.2)
|   |   |   F10 > 4.300000 : 1 (4.0/2.2)
|   F1 > 0.000000 :
|   |   F10 <= 2.600000 :
|   |   |   F1 <= 1.160000 : 1 (5.0/1.2)
|   |   |   F1 > 1.160000 : 0 (4.0/1.2)
|   |   F10 > 2.600000 : 1 (39.0/1.4)
