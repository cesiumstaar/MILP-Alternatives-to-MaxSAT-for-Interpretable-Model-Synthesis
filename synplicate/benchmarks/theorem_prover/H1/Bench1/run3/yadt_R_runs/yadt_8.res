YaDT version 2.2.0 (c) S. Ruggieri 2002-2019

MAJORITY CLASS value: 1 (2 classes)
MAJORITY MISCLASSIFICATION on training: 44.5946% of 148 cases. 

MISCLASSIFICATION on training: 0.675676% of 148 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	65	1	0
0	0	82	1

MISCLASSIFICATION on test file: 19.0669% of 493 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	179	83	0
0	11	220	1

DECISION TREE (size=13)

F1 <= 0.000000 :
|   F1 <= -0.290000 : 0 (33.0/1.4)
|   F1 > -0.290000 :
|   |   F10 <= 4.300000 : 0 (10.0/1.3)
|   |   F10 > 4.300000 : 1 (5.0/2.3)
F1 > 0.000000 :
|   F10 <= 3.000000 :
|   |   F1 <= 1.330000 :
|   |   |   F10 <= 1.600000 : 0 (7.0/1.3)
|   |   |   F10 > 1.600000 : 1 (10.0/1.3)
|   |   F1 > 1.330000 : 0 (15.0/1.3)
|   F10 > 3.000000 : 1 (68.0/1.4)
