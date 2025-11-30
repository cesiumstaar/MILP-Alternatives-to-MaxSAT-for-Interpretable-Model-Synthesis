YaDT version 2.2.0 (c) S. Ruggieri 2002-2019

MAJORITY CLASS value: 0 (2 classes)
MAJORITY MISCLASSIFICATION on training: 31.7568% of 148 cases. 

MISCLASSIFICATION on training: 0.675676% of 148 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	100	1	0
0	0	47	1

MISCLASSIFICATION on test file: 12.8866% of 970 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	603	100	0
0	25	242	1

DECISION TREE (size=19)

F1 <= -0.130000 : 0 (58.0/1.4)
F1 > -0.130000 :
|   F10 <= 2.700000 :
|   |   F10 <= 1.400000 : 0 (21.0/1.3)
|   |   F10 > 1.400000 :
|   |   |   F1 <= 1.240000 :
|   |   |   |   F1 <= 0.350000 : 0 (7.0/1.3)
|   |   |   |   F1 > 0.350000 :
|   |   |   |   |   F10 <= 1.900000 :
|   |   |   |   |   |   F1 <= 0.870000 : 1 (2.0/1.0)
|   |   |   |   |   |   F1 > 0.870000 : 0 (2.0/1.0)
|   |   |   |   |   F10 > 1.900000 : 1 (7.0/1.3)
|   |   |   F1 > 1.240000 : 0 (10.0/1.3)
|   F10 > 2.700000 :
|   |   F1 <= 1.870000 : 1 (30.0/1.4)
|   |   F1 > 1.870000 :
|   |   |   F10 <= 3.400000 : 0 (2.0/1.0)
|   |   |   F10 > 3.400000 : 1 (9.0/2.4)
