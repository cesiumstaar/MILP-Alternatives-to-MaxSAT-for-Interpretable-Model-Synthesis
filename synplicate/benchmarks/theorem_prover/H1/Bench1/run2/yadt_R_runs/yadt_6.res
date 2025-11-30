YaDT version 2.2.0 (c) S. Ruggieri 2002-2019

MAJORITY CLASS value: 1 (2 classes)
MAJORITY MISCLASSIFICATION on training: 48.2759% of 116 cases. 

MISCLASSIFICATION on training: 1.72414% of 116 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	55	1	0
0	1	59	1

MISCLASSIFICATION on test file: 19.3622% of 439 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	183	74	0
0	11	171	1

DECISION TREE (size=9)

F1 <= 0.130000 : 0 (37.0/2.6)
F1 > 0.130000 :
|   F10 <= 2.800000 :
|   |   F10 <= 2.000000 : 0 (13.0/1.3)
|   |   F10 > 2.000000 :
|   |   |   F1 <= 1.300000 : 1 (11.0/2.5)
|   |   |   F1 > 1.300000 : 0 (6.0/1.2)
|   F10 > 2.800000 : 1 (49.0/1.4)
