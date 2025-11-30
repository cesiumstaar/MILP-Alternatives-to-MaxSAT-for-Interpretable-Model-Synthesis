YaDT version 2.2.0 (c) S. Ruggieri 2002-2019

MAJORITY CLASS value: 0 (2 classes)
MAJORITY MISCLASSIFICATION on training: 37.1212% of 132 cases. 

MISCLASSIFICATION on training: 1.51515% of 132 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	81	2	0
0	0	49	1

MISCLASSIFICATION on test file: 18.0645% of 465 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	264	52	0
0	32	117	1

DECISION TREE (size=11)

F1 <= 0.120000 : 0 (45.0/1.4)
F1 > 0.120000 :
|   F10 <= 1.600000 : 0 (27.0/1.4)
|   F10 > 1.600000 :
|   |   F1 <= 1.630000 :
|   |   |   F10 <= 2.500000 :
|   |   |   |   F1 <= 1.220000 : 1 (12.0/3.6)
|   |   |   |   F1 > 1.220000 : 0 (3.0/1.1)
|   |   |   F10 > 2.500000 : 1 (39.0/1.4)
|   |   F1 > 1.630000 : 0 (6.0/1.2)
