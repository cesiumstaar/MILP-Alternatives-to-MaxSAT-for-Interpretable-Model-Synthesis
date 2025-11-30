YaDT version 2.2.0 (c) S. Ruggieri 2002-2019

MAJORITY CLASS value: 0 (2 classes)
MAJORITY MISCLASSIFICATION on training: 41% of 100 cases. 

MISCLASSIFICATION on training: 1% of 100 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	59	0	0
0	1	40	1

MISCLASSIFICATION on test file: 15.0273% of 366 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	200	44	0
0	11	111	1

DECISION TREE (size=11)

F10 <= 1.400000 : 0 (29.0/1.4)
F10 > 1.400000 :
|   F1 <= -0.030000 : 0 (21.0/2.5)
|   F1 > -0.030000 :
|   |   F10 <= 2.900000 :
|   |   |   F1 <= 1.240000 :
|   |   |   |   F1 <= 0.320000 : 0 (2.0/1.0)
|   |   |   |   F1 > 0.320000 : 1 (13.0/1.3)
|   |   |   F1 > 1.240000 : 0 (8.0/1.3)
|   |   F10 > 2.900000 : 1 (27.0/1.4)
