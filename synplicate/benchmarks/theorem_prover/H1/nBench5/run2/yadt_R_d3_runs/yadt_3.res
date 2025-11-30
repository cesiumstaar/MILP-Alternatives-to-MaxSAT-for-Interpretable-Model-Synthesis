YaDT version 2.2.0 (c) S. Ruggieri 2002-2019

MAJORITY CLASS value: 1 (2 classes)
MAJORITY MISCLASSIFICATION on training: 38.2353% of 68 cases. 

MISCLASSIFICATION on training: 4.41176% of 68 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	23	3	0
0	0	42	1

MISCLASSIFICATION on test file: 25.8741% of 286 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	84	68	0
0	6	128	1

DECISION TREE (size=9)

F1 <= 0.220000 :
|   F10 <= 5.000000 : 0 (17.0/1.3)
|   F10 > 5.000000 : 1 (3.0/2.1)
F1 > 0.220000 :
|   F10 <= 2.500000 :
|   |   F10 <= 1.500000 : 0 (6.0/1.2)
|   |   F10 > 1.500000 : 1 (6.0/3.3)
|   F10 > 2.500000 : 1 (36.0/1.4)
