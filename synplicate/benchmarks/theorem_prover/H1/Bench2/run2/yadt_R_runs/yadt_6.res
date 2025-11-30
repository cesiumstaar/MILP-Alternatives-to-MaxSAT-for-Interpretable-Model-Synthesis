YaDT version 2.2.0 (c) S. Ruggieri 2002-2019

MAJORITY CLASS value: 0 (2 classes)
MAJORITY MISCLASSIFICATION on training: 34.4828% of 116 cases. 

MISCLASSIFICATION on training: 4.31034% of 116 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	74	2	0
0	3	37	1

MISCLASSIFICATION on test file: 12.3916% of 807 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	515	77	0
0	23	192	1

DECISION TREE (size=15)

F10 <= 2.000000 : 0 (50.0/5.0)
F10 > 2.000000 :
|   F1 <= -0.340000 : 0 (17.0/1.3)
|   F1 > -0.340000 :
|   |   F10 <= 3.700000 :
|   |   |   F1 <= 2.210000 :
|   |   |   |   F1 <= 0.150000 : 0 (3.0/1.1)
|   |   |   |   F1 > 0.150000 :
|   |   |   |   |   F10 <= 2.700000 :
|   |   |   |   |   |   F1 <= 1.520000 : 1 (9.0/2.4)
|   |   |   |   |   |   F1 > 1.520000 : 0 (2.0/1.0)
|   |   |   |   |   F10 > 2.700000 : 1 (13.0/1.3)
|   |   |   F1 > 2.210000 : 0 (5.0/1.2)
|   |   F10 > 3.700000 : 1 (17.0/2.5)
