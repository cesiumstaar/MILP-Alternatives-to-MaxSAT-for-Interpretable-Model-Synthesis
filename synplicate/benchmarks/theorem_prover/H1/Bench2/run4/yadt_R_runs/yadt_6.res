YaDT version 2.2.0 (c) S. Ruggieri 2002-2019

MAJORITY CLASS value: 0 (2 classes)
MAJORITY MISCLASSIFICATION on training: 48.2759% of 116 cases. 

MISCLASSIFICATION on training: 1.72414% of 116 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	59	1	0
0	1	55	1

MISCLASSIFICATION on test file: 18.4244% of 787 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	398	126	0
0	19	244	1

DECISION TREE (size=13)

F1 <= -0.220000 : 0 (40.0/1.4)
F1 > -0.220000 :
|   F10 <= 2.700000 :
|   |   F1 <= 1.210000 :
|   |   |   F10 <= 1.300000 : 0 (2.0/1.0)
|   |   |   F10 > 1.300000 : 1 (12.0/2.5)
|   |   F1 > 1.210000 : 0 (13.0/1.3)
|   F10 > 2.700000 :
|   |   F1 <= 2.210000 : 1 (40.0/1.4)
|   |   F1 > 2.210000 :
|   |   |   F10 <= 4.400000 : 0 (5.0/2.3)
|   |   |   F10 > 4.400000 : 1 (4.0/1.2)
