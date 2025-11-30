YaDT version 2.2.0 (c) S. Ruggieri 2002-2019

MAJORITY CLASS value: 1 (2 classes)
MAJORITY MISCLASSIFICATION on training: 38.2353% of 68 cases. 

MISCLASSIFICATION on training: 1.47059% of 68 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	25	1	0
0	0	42	1

MISCLASSIFICATION on test file: 21.3287% of 286 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	97	55	0
0	6	128	1

DECISION TREE (size=11)

F1 <= 0.220000 :
|   F10 <= 5.000000 : 0 (17.0/1.3)
|   F10 > 5.000000 : 1 (3.0/2.1)
F1 > 0.220000 :
|   F10 <= 2.500000 :
|   |   F10 <= 1.500000 : 0 (6.0/1.2)
|   |   F10 > 1.500000 :
|   |   |   F1 <= 1.000000 : 1 (4.0/1.2)
|   |   |   F1 > 1.000000 : 0 (2.0/1.0)
|   F10 > 2.500000 : 1 (36.0/1.4)
