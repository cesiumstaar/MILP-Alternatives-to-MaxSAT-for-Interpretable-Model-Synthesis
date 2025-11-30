YaDT version 2.2.0 (c) S. Ruggieri 2002-2019

MAJORITY CLASS value: 0 (2 classes)
MAJORITY MISCLASSIFICATION on training: 47.2973% of 148 cases. 

MISCLASSIFICATION on training: 0% of 148 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	78	0	0
0	0	70	1

MISCLASSIFICATION on test file: 17.7189% of 491 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	214	85	0
0	2	190	1

DECISION TREE (size=15)

F1 <= -0.100000 : 0 (48.0/1.4)
F1 > -0.100000 :
|   F10 <= 2.700000 :
|   |   F1 <= 1.260000 :
|   |   |   F10 <= 1.500000 : 0 (8.0/1.3)
|   |   |   F10 > 1.500000 :
|   |   |   |   F10 <= 2.200000 :
|   |   |   |   |   F1 <= 0.260000 : 0 (3.0/1.1)
|   |   |   |   |   F1 > 0.260000 :
|   |   |   |   |   |   F1 <= 1.190000 : 1 (5.0/1.2)
|   |   |   |   |   |   F1 > 1.190000 : 0 (2.0/1.0)
|   |   |   |   F10 > 2.200000 : 1 (8.0/1.3)
|   |   F1 > 1.260000 : 0 (17.0/1.3)
|   F10 > 2.700000 : 1 (57.0/1.4)
