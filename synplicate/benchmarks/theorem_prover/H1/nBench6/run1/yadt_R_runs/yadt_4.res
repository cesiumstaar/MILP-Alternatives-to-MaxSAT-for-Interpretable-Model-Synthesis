YaDT version 2.2.0 (c) S. Ruggieri 2002-2019

MAJORITY CLASS value: 0 (2 classes)
MAJORITY MISCLASSIFICATION on training: 42.8571% of 84 cases. 

MISCLASSIFICATION on training: 1.19048% of 84 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	48	0	0
0	1	35	1

MISCLASSIFICATION on test file: 14.9725% of 728 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	409	88	0
0	21	210	1

DECISION TREE (size=15)

F1 <= 0.110000 : 0 (32.0/2.6)
F1 > 0.110000 :
|   F10 <= 2.500000 :
|   |   F10 <= 1.500000 : 0 (11.0/1.3)
|   |   F10 > 1.500000 :
|   |   |   F1 <= 0.450000 : 0 (2.0/1.0)
|   |   |   F1 > 0.450000 :
|   |   |   |   F1 <= 1.300000 : 1 (3.0/1.1)
|   |   |   |   F1 > 1.300000 : 0 (2.0/1.0)
|   F10 > 2.500000 :
|   |   F1 <= 1.300000 : 1 (27.0/1.4)
|   |   F1 > 1.300000 :
|   |   |   F10 <= 3.400000 : 0 (2.0/1.0)
|   |   |   F10 > 3.400000 : 1 (5.0/1.2)
