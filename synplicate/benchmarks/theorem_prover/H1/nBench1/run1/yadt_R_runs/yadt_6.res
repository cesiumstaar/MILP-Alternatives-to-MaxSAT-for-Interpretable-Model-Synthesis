YaDT version 2.2.0 (c) S. Ruggieri 2002-2019

MAJORITY CLASS value: 0 (2 classes)
MAJORITY MISCLASSIFICATION on training: 38.7931% of 116 cases. 

MISCLASSIFICATION on training: 1.72414% of 116 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	70	1	0
0	1	44	1

MISCLASSIFICATION on test file: 14.8585% of 424 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	247	49	0
0	14	114	1

DECISION TREE (size=11)

F1 <= 0.180000 : 0 (43.0/2.6)
F1 > 0.180000 :
|   F10 <= 2.500000 :
|   |   F1 <= 0.900000 :
|   |   |   F10 <= 1.300000 : 0 (6.0/1.2)
|   |   |   F10 > 1.300000 :
|   |   |   |   F1 <= 0.440000 : 0 (3.0/1.1)
|   |   |   |   F1 > 0.440000 : 1 (10.0/2.4)
|   |   F1 > 0.900000 : 0 (19.0/1.3)
|   F10 > 2.500000 : 1 (35.0/1.4)
