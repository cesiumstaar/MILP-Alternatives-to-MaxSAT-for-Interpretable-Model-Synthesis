YaDT version 2.2.0 (c) S. Ruggieri 2002-2019

MAJORITY CLASS value: 0 (2 classes)
MAJORITY MISCLASSIFICATION on training: 42.5676% of 148 cases. 

MISCLASSIFICATION on training: 1.35135% of 148 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	85	0	0
0	2	61	1

MISCLASSIFICATION on test file: 13.0011% of 923 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	527	102	0
0	18	276	1

DECISION TREE (size=15)

F1 <= 0.100000 : 0 (57.0/2.6)
F1 > 0.100000 :
|   F10 <= 2.300000 :
|   |   F10 <= 1.200000 : 0 (12.0/1.3)
|   |   F10 > 1.200000 :
|   |   |   F1 <= 1.210000 :
|   |   |   |   F1 <= 0.320000 : 0 (4.0/1.2)
|   |   |   |   F1 > 0.320000 : 1 (5.0/1.2)
|   |   |   F1 > 1.210000 : 0 (7.0/1.3)
|   F10 > 2.300000 :
|   |   F1 <= 1.600000 : 1 (46.0/1.4)
|   |   F1 > 1.600000 :
|   |   |   F10 <= 3.900000 : 0 (7.0/2.4)
|   |   |   F10 > 3.900000 : 1 (10.0/1.3)
