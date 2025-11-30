YaDT version 2.2.0 (c) S. Ruggieri 2002-2019

MAJORITY CLASS value: 0 (2 classes)
MAJORITY MISCLASSIFICATION on training: 46.087% of 115 cases. 

MISCLASSIFICATION on training: 1.73913% of 115 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	61	1	0
0	1	52	1

MISCLASSIFICATION on test file: 19.1283% of 413 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	193	72	0
0	7	141	1

DECISION TREE (size=15)

F1 <= 0.290000 :
|   F10 <= 5.000000 : 0 (35.0/1.4)
|   F10 > 5.000000 : 1 (3.0/2.1)
F1 > 0.290000 :
|   F10 <= 2.600000 :
|   |   F10 <= 1.800000 : 0 (20.0/2.5)
|   |   F10 > 1.800000 :
|   |   |   F1 <= 1.470000 :
|   |   |   |   F10 <= 2.100000 :
|   |   |   |   |   F10 <= 2.000000 : 1 (2.0/1.0)
|   |   |   |   |   F10 > 2.000000 : 0 (2.0/1.0)
|   |   |   |   F10 > 2.100000 : 1 (6.0/1.2)
|   |   |   F1 > 1.470000 : 0 (5.0/1.2)
|   F10 > 2.600000 : 1 (42.0/1.4)
