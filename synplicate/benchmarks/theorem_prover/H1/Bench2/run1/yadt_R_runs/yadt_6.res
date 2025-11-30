YaDT version 2.2.0 (c) S. Ruggieri 2002-2019

MAJORITY CLASS value: 0 (2 classes)
MAJORITY MISCLASSIFICATION on training: 45.6897% of 116 cases. 

MISCLASSIFICATION on training: 1.72414% of 116 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	61	2	0
0	0	53	1

MISCLASSIFICATION on test file: 14.881% of 840 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	447	112	0
0	13	268	1

DECISION TREE (size=17)

F1 <= -0.360000 : 0 (30.0/1.4)
F1 > -0.360000 :
|   F10 <= 2.700000 :
|   |   F1 <= 0.920000 :
|   |   |   F1 <= 0.400000 : 0 (6.0/1.2)
|   |   |   F1 > 0.400000 : 1 (8.0/2.4)
|   |   F1 > 0.920000 : 0 (17.0/1.3)
|   F10 > 2.700000 :
|   |   F10 <= 4.700000 :
|   |   |   F1 <= -0.120000 : 0 (4.0/1.2)
|   |   |   F1 > -0.120000 :
|   |   |   |   F1 <= 1.880000 : 1 (26.0/1.3)
|   |   |   |   F1 > 1.880000 :
|   |   |   |   |   F10 <= 3.700000 : 0 (4.0/1.2)
|   |   |   |   |   F10 > 3.700000 : 1 (4.0/2.2)
|   |   F10 > 4.700000 : 1 (17.0/1.3)
