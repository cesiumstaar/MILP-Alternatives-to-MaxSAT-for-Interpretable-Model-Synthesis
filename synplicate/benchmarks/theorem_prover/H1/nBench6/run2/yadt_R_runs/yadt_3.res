YaDT version 2.2.0 (c) S. Ruggieri 2002-2019

MAJORITY CLASS value: 0 (2 classes)
MAJORITY MISCLASSIFICATION on training: 39.7059% of 68 cases. 

MISCLASSIFICATION on training: 1.47059% of 68 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	40	1	0
0	0	27	1

MISCLASSIFICATION on test file: 13.229% of 703 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	409	84	0
0	9	201	1

DECISION TREE (size=15)

F1 <= -0.760000 : 0 (20.0/1.3)
F1 > -0.760000 :
|   F10 <= 2.600000 :
|   |   F10 <= 1.400000 : 0 (9.0/1.3)
|   |   F10 > 1.400000 :
|   |   |   F1 <= 0.350000 : 0 (6.0/1.2)
|   |   |   F1 > 0.350000 :
|   |   |   |   F1 <= 0.870000 : 1 (5.0/1.2)
|   |   |   |   F1 > 0.870000 :
|   |   |   |   |   F1 <= 1.240000 :
|   |   |   |   |   |   F10 <= 1.900000 : 0 (2.0/1.0)
|   |   |   |   |   |   F10 > 1.900000 : 1 (2.0/1.0)
|   |   |   |   |   F1 > 1.240000 : 0 (3.0/1.1)
|   F10 > 2.600000 : 1 (21.0/2.5)
