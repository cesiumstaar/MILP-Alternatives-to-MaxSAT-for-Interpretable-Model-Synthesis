YaDT version 2.2.0 (c) S. Ruggieri 2002-2019

MAJORITY CLASS value: 0 (2 classes)
MAJORITY MISCLASSIFICATION on training: 41.4634% of 164 cases. 

MISCLASSIFICATION on training: 0.609756% of 164 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	96	0	0
0	1	67	1

MISCLASSIFICATION on test file: 12.5382% of 981 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	554	113	0
0	10	304	1

DECISION TREE (size=19)

F1 <= 0.100000 :
|   F10 <= 6.000000 : 0 (61.0/1.4)
|   F10 > 6.000000 :
|   |   F1 <= -0.680000 : 0 (4.0/1.2)
|   |   F1 > -0.680000 : 1 (2.0/1.0)
F1 > 0.100000 :
|   F10 <= 2.300000 :
|   |   F10 <= 1.200000 : 0 (12.0/1.3)
|   |   F10 > 1.200000 :
|   |   |   F1 <= 1.210000 :
|   |   |   |   F1 <= 0.320000 : 0 (4.0/1.2)
|   |   |   |   F1 > 0.320000 : 1 (5.0/1.2)
|   |   |   F1 > 1.210000 : 0 (7.0/1.3)
|   F10 > 2.300000 :
|   |   F1 <= 1.600000 : 1 (49.0/1.4)
|   |   F1 > 1.600000 :
|   |   |   F10 <= 3.900000 : 0 (9.0/2.4)
|   |   |   F10 > 3.900000 : 1 (11.0/1.3)
