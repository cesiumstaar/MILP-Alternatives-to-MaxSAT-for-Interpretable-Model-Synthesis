YaDT version 2.2.0 (c) S. Ruggieri 2002-2019

MAJORITY CLASS value: 0 (2 classes)
MAJORITY MISCLASSIFICATION on training: 38.0952% of 84 cases. 

MISCLASSIFICATION on training: 1.19048% of 84 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	52	0	0
0	1	31	1

MISCLASSIFICATION on test file: 13.8889% of 756 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	435	73	0
0	32	216	1

DECISION TREE (size=11)

F1 <= -0.550000 : 0 (30.0/1.4)
F1 > -0.550000 :
|   F10 <= 1.600000 : 0 (15.0/1.3)
|   F10 > 1.600000 :
|   |   F10 <= 3.400000 :
|   |   |   F1 <= 0.290000 : 0 (3.0/1.1)
|   |   |   F1 > 0.290000 :
|   |   |   |   F1 <= 1.240000 : 1 (9.0/1.3)
|   |   |   |   F1 > 1.240000 : 0 (5.0/2.3)
|   |   F10 > 3.400000 : 1 (22.0/1.3)
