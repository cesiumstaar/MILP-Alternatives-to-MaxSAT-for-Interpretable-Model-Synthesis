YaDT version 2.2.0 (c) S. Ruggieri 2002-2019

MAJORITY CLASS value: 0 (2 classes)
MAJORITY MISCLASSIFICATION on training: 36.5854% of 164 cases. 

MISCLASSIFICATION on training: 1.82927% of 164 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	103	1	0
0	2	58	1

MISCLASSIFICATION on test file: 13.4796% of 957 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	567	103	0
0	26	261	1

DECISION TREE (size=15)

F1 <= -0.300000 : 0 (50.0/1.4)
F1 > -0.300000 :
|   F10 <= 1.800000 : 0 (33.0/2.6)
|   F10 > 1.800000 :
|   |   F10 <= 4.000000 :
|   |   |   F1 <= 1.850000 :
|   |   |   |   F1 <= 0.230000 : 0 (5.0/1.2)
|   |   |   |   F1 > 0.230000 :
|   |   |   |   |   F10 <= 2.700000 :
|   |   |   |   |   |   F1 <= 0.900000 : 1 (8.0/1.3)
|   |   |   |   |   |   F1 > 0.900000 : 0 (4.0/1.2)
|   |   |   |   |   F10 > 2.700000 : 1 (20.0/1.3)
|   |   |   F1 > 1.850000 : 0 (13.0/2.5)
|   |   F10 > 4.000000 : 1 (31.0/2.6)
