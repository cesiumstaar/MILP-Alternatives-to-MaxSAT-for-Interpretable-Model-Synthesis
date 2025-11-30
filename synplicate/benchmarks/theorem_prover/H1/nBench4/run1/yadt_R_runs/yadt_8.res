YaDT version 2.2.0 (c) S. Ruggieri 2002-2019

MAJORITY CLASS value: 0 (2 classes)
MAJORITY MISCLASSIFICATION on training: 35.8108% of 148 cases. 

MISCLASSIFICATION on training: 2.02703% of 148 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	94	1	0
0	2	51	1

MISCLASSIFICATION on test file: 13.3477% of 929 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	552	99	0
0	25	253	1

DECISION TREE (size=15)

F1 <= -0.300000 : 0 (48.0/1.4)
F1 > -0.300000 :
|   F10 <= 1.800000 : 0 (30.0/2.6)
|   F10 > 1.800000 :
|   |   F10 <= 4.000000 :
|   |   |   F1 <= 1.850000 :
|   |   |   |   F1 <= 0.230000 : 0 (5.0/1.2)
|   |   |   |   F1 > 0.230000 :
|   |   |   |   |   F10 <= 2.700000 :
|   |   |   |   |   |   F1 <= 0.900000 : 1 (7.0/1.3)
|   |   |   |   |   |   F1 > 0.900000 : 0 (3.0/1.1)
|   |   |   |   |   F10 > 2.700000 : 1 (17.0/1.3)
|   |   |   F1 > 1.850000 : 0 (10.0/2.4)
|   |   F10 > 4.000000 : 1 (28.0/2.6)
