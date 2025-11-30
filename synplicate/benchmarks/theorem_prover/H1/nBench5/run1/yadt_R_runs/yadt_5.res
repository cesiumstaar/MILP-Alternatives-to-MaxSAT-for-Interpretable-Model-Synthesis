YaDT version 2.2.0 (c) S. Ruggieri 2002-2019

MAJORITY CLASS value: 0 (2 classes)
MAJORITY MISCLASSIFICATION on training: 44% of 100 cases. 

MISCLASSIFICATION on training: 2% of 100 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	55	1	0
0	1	43	1

MISCLASSIFICATION on test file: 14.5455% of 385 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	207	47	0
0	9	122	1

DECISION TREE (size=11)

F1 <= 0.170000 : 0 (34.0/2.6)
F1 > 0.170000 :
|   F10 <= 2.800000 :
|   |   F10 <= 1.600000 : 0 (13.0/1.3)
|   |   F10 > 1.600000 :
|   |   |   F1 <= 1.190000 :
|   |   |   |   F1 <= 0.280000 : 0 (3.0/1.1)
|   |   |   |   F1 > 0.280000 : 1 (10.0/2.4)
|   |   |   F1 > 1.190000 : 0 (6.0/1.2)
|   F10 > 2.800000 : 1 (34.0/1.4)
