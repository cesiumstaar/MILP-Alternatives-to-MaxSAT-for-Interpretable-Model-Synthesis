YaDT version 2.2.0 (c) S. Ruggieri 2002-2019

MAJORITY CLASS value: 0 (2 classes)
MAJORITY MISCLASSIFICATION on training: 32.3171% of 164 cases. 

MISCLASSIFICATION on training: 1.82927% of 164 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	110	1	0
0	2	51	1

MISCLASSIFICATION on test file: 15.2174% of 966 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	599	109	0
0	38	220	1

DECISION TREE (size=13)

F1 <= 0.160000 : 0 (62.0/2.6)
F1 > 0.160000 :
|   F10 <= 2.300000 :
|   |   F1 <= 0.780000 :
|   |   |   F10 <= 1.100000 : 0 (8.0/1.3)
|   |   |   F10 > 1.100000 : 1 (9.0/2.4)
|   |   F1 > 0.780000 : 0 (31.0/1.4)
|   F10 > 2.300000 :
|   |   F1 <= 1.660000 : 1 (37.0/1.4)
|   |   F1 > 1.660000 :
|   |   |   F10 <= 3.900000 : 0 (11.0/2.5)
|   |   |   F10 > 3.900000 : 1 (6.0/1.2)
