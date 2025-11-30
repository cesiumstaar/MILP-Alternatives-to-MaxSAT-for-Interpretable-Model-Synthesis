YaDT version 2.2.0 (c) S. Ruggieri 2002-2019

MAJORITY CLASS value: 1 (2 classes)
MAJORITY MISCLASSIFICATION on training: 41% of 100 cases. 

MISCLASSIFICATION on training: 0% of 100 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	41	0	0
0	0	59	1

MISCLASSIFICATION on test file: 21.0383% of 366 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	133	71	0
0	6	156	1

DECISION TREE (size=13)

F1 <= -0.200000 : 0 (21.0/1.3)
F1 > -0.200000 :
|   F10 <= 1.500000 : 0 (9.0/1.3)
|   F10 > 1.500000 :
|   |   F1 <= 0.220000 :
|   |   |   F10 <= 4.800000 : 0 (6.0/1.2)
|   |   |   F10 > 4.800000 : 1 (4.0/1.2)
|   |   F1 > 0.220000 :
|   |   |   F10 <= 2.700000 :
|   |   |   |   F1 <= 1.430000 : 1 (9.0/1.3)
|   |   |   |   F1 > 1.430000 : 0 (5.0/1.2)
|   |   |   F10 > 2.700000 : 1 (46.0/1.4)
