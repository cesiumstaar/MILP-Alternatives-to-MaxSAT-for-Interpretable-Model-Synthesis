YaDT version 2.2.0 (c) S. Ruggieri 2002-2019

MAJORITY CLASS value: 0 (2 classes)
MAJORITY MISCLASSIFICATION on training: 30% of 100 cases. 

MISCLASSIFICATION on training: 2% of 100 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	68	2	0
0	0	30	1

MISCLASSIFICATION on test file: 13.7278% of 845 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	511	97	0
0	19	218	1

DECISION TREE (size=15)

F1 <= -0.130000 : 0 (38.0/1.4)
F1 > -0.130000 :
|   F10 <= 2.700000 :
|   |   F10 <= 1.400000 : 0 (15.0/1.3)
|   |   F10 > 1.400000 :
|   |   |   F1 <= 1.240000 :
|   |   |   |   F1 <= 0.350000 : 0 (6.0/1.2)
|   |   |   |   F1 > 0.350000 :
|   |   |   |   |   F10 <= 1.900000 :
|   |   |   |   |   |   F1 <= 0.870000 : 1 (2.0/1.0)
|   |   |   |   |   |   F1 > 0.870000 : 0 (2.0/1.0)
|   |   |   |   |   F10 > 1.900000 : 1 (6.0/1.2)
|   |   |   F1 > 1.240000 : 0 (7.0/1.3)
|   F10 > 2.700000 : 1 (24.0/3.7)
