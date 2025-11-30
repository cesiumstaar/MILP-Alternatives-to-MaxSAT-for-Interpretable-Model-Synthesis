YaDT version 2.2.0 (c) S. Ruggieri 2002-2019

MAJORITY CLASS value: 0 (2 classes)
MAJORITY MISCLASSIFICATION on training: 42.4242% of 132 cases. 

MISCLASSIFICATION on training: 0.757576% of 132 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	76	0	0
0	1	55	1

MISCLASSIFICATION on test file: 12.7128% of 881 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	500	96	0
0	16	269	1

DECISION TREE (size=15)

F1 <= 0.100000 : 0 (51.0/1.4)
F1 > 0.100000 :
|   F10 <= 2.300000 :
|   |   F10 <= 1.200000 : 0 (10.0/1.3)
|   |   F10 > 1.200000 :
|   |   |   F1 <= 1.210000 :
|   |   |   |   F1 <= 0.320000 : 0 (4.0/1.2)
|   |   |   |   F1 > 0.320000 : 1 (5.0/1.2)
|   |   |   F1 > 1.210000 : 0 (6.0/1.2)
|   F10 > 2.300000 :
|   |   F1 <= 1.600000 : 1 (41.0/1.4)
|   |   F1 > 1.600000 :
|   |   |   F10 <= 3.900000 : 0 (6.0/2.3)
|   |   |   F10 > 3.900000 : 1 (9.0/1.3)
