YaDT version 2.2.0 (c) S. Ruggieri 2002-2019

MAJORITY CLASS value: 0 (2 classes)
MAJORITY MISCLASSIFICATION on training: 29.0541% of 148 cases. 

MISCLASSIFICATION on training: 5.40541% of 148 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	101	4	0
0	4	39	1

MISCLASSIFICATION on test file: 15.7952% of 918 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	566	122	0
0	23	207	1

DECISION TREE (size=11)

F1 <= -0.480000 : 0 (43.0/1.4)
F1 > -0.480000 :
|   F10 <= 2.000000 : 0 (47.0/6.1)
|   F10 > 2.000000 :
|   |   F10 <= 3.200000 :
|   |   |   F1 <= 1.660000 :
|   |   |   |   F1 <= 0.210000 : 0 (6.0/1.2)
|   |   |   |   F1 > 0.210000 : 1 (14.0/1.3)
|   |   |   F1 > 1.660000 : 0 (9.0/1.3)
|   |   F10 > 3.200000 : 1 (29.0/6.0)
