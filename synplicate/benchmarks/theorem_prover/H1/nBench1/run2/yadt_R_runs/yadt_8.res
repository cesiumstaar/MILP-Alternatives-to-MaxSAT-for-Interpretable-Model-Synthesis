YaDT version 2.2.0 (c) S. Ruggieri 2002-2019

MAJORITY CLASS value: 0 (2 classes)
MAJORITY MISCLASSIFICATION on training: 39.1892% of 148 cases. 

MISCLASSIFICATION on training: 1.35135% of 148 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	90	0	0
0	2	56	1

MISCLASSIFICATION on test file: 15.9919% of 494 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	282	52	0
0	27	133	1

DECISION TREE (size=17)

F1 <= 0.200000 :
|   F10 <= 3.500000 : 0 (42.0/1.4)
|   F10 > 3.500000 :
|   |   F10 <= 4.900000 :
|   |   |   F1 <= -0.010000 : 0 (10.0/1.3)
|   |   |   F1 > -0.010000 : 1 (2.0/1.0)
|   |   F10 > 4.900000 : 1 (2.0/1.0)
F1 > 0.200000 :
|   F10 <= 1.500000 : 0 (23.0/1.3)
|   F10 > 1.500000 :
|   |   F1 <= 0.990000 : 1 (33.0/1.4)
|   |   F1 > 0.990000 :
|   |   |   F10 <= 2.800000 : 0 (14.0/2.5)
|   |   |   F10 > 2.800000 :
|   |   |   |   F1 <= 1.870000 : 1 (19.0/1.3)
|   |   |   |   F1 > 1.870000 : 0 (3.0/2.1)
