YaDT version 2.2.0 (c) S. Ruggieri 2002-2019

MAJORITY CLASS value: 0 (2 classes)
MAJORITY MISCLASSIFICATION on training: 44.1176% of 68 cases. 

MISCLASSIFICATION on training: 0% of 68 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	38	0	0
0	0	30	1

MISCLASSIFICATION on test file: 14.527% of 296 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	167	34	0
0	9	86	1

DECISION TREE (size=11)

F1 <= -0.200000 : 0 (18.0/1.3)
F1 > -0.200000 :
|   F10 <= 2.500000 :
|   |   F10 <= 1.300000 : 0 (10.0/1.3)
|   |   F10 > 1.300000 :
|   |   |   F1 <= 0.800000 :
|   |   |   |   F1 <= 0.440000 : 0 (3.0/1.1)
|   |   |   |   F1 > 0.440000 : 1 (8.0/1.3)
|   |   |   F1 > 0.800000 : 0 (7.0/1.3)
|   F10 > 2.500000 : 1 (22.0/1.3)
